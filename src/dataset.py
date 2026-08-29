import tqdm
import numpy as np
import torch
import os
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import random
import math

# Default time interval bucketing thresholds (in seconds)
# 0, 1min, 5min, 15min, 1h, 6h, 1d, 3d, 1w, 1mo, 3mo, 6mo, 1yr, >1yr
DEFAULT_TIME_THRESHOLDS = [0, 60, 300, 900, 3600, 21600, 86400, 259200, 604800, 2592000, 7776000, 15552000, 31104000]

def time_to_bucket(time_delta, thresholds=DEFAULT_TIME_THRESHOLDS):
    """Convert time delta (seconds) to bucket ID."""
    for i, thresh in enumerate(thresholds):
        if time_delta <= thresh:
            return i
    return len(thresholds)

def get_time_seqs(time_file, thresholds=DEFAULT_TIME_THRESHOLDS, prebucketed=False):
    """Load time interval sequences from file.
    If prebucketed=True, values are already discrete bucket IDs (e.g. TASIF absolute time spans).
    """
    lines = open(time_file).readlines()
    time_seq = []
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) < 2:
            continue
        times = [int(t) for t in parts[1].split()]
        if not prebucketed:
            times = [time_to_bucket(t, thresholds) for t in times]
        time_seq.append(times)
    return time_seq, thresholds

class RecDataset(Dataset):
    def __init__(self, args, user_seq, time_seq=None, test_neg_items=None, data_type='train'):
        self.args = args
        self.user_seq = []
        self.time_seq = []
        self.max_len = args.max_seq_length
        self.user_ids = []
        self.contrastive_learning = args.model_type.lower() in ['fearec', 'duorec', 'slime4rec']
        self.data_type = data_type
        self.has_time = time_seq is not None

        if self.data_type=='train':
            for user, seq in enumerate(user_seq):
                input_ids = seq[-(self.max_len + 2):-2]
                if self.has_time:
                    time_ids = time_seq[user][-(self.max_len + 2):-2]
                for i in range(len(input_ids)):
                    self.user_seq.append(input_ids[:i + 1])
                    if self.has_time:
                        self.time_seq.append(time_ids[:i + 1])
                    self.user_ids.append(user)
        elif self.data_type=='valid':
            for i, sequence in enumerate(user_seq):
                self.user_seq.append(sequence[:-1])
                if self.has_time:
                    self.time_seq.append(time_seq[i][:-1])
        else:
            self.user_seq = user_seq
            if self.has_time:
                self.time_seq = time_seq

        self.test_neg_items = test_neg_items

        if self.contrastive_learning and self.data_type=='train':
            if os.path.exists(args.same_target_path):
                self.same_target_index = np.load(args.same_target_path, allow_pickle=True)
            else:
                print("Start making same_target_index for contrastive learning")
                self.same_target_index = self.get_same_target_index()
                self.same_target_index = np.array(self.same_target_index, dtype=object)
                np.save(args.same_target_path, self.same_target_index)

    def get_same_target_index(self):
        num_items = max([max(v) for v in self.user_seq]) + 2
        same_target_index = [[] for _ in range(num_items)]
        
        user_seq = self.user_seq[:]
        tmp_user_seq = []
        for i in tqdm.tqdm(range(1, num_items)):
            for j in range(len(user_seq)):
                if user_seq[j][-1] == i:
                    same_target_index[i].append(user_seq[j])
                else:
                    tmp_user_seq.append(user_seq[j])
            user_seq = tmp_user_seq
            tmp_user_seq = []

        return same_target_index

    def __len__(self):
        return len(self.user_seq)

    def __getitem__(self, index):
        items = self.user_seq[index]
        input_ids = items[:-1]
        answer = items[-1]

        seq_set = set(items)
        neg_answer = neg_sample(seq_set, self.args.item_size)

        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        input_ids = input_ids[-self.max_len:]
        assert len(input_ids) == self.max_len

        # Time interval sequence: same padding/truncation as items
        if self.has_time:
            time_ids = self.time_seq[index]
            time_input = time_ids[:-1]
            time_input = [0] * pad_len + time_input
            time_input = time_input[-self.max_len:]
            assert len(time_input) == self.max_len
        else:
            time_input = [0] * self.max_len

        if self.data_type in ['valid', 'test']:
            cur_tensors = (
                torch.tensor(index, dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.zeros(0, dtype=torch.long), # not used
                torch.zeros(0, dtype=torch.long), # not used
                torch.tensor(time_input, dtype=torch.long), # time span ids
            )

        elif self.contrastive_learning:
            sem_augs = self.same_target_index[answer]
            sem_aug = random.choice(sem_augs)
            keep_random = False
            for i in range(len(sem_augs)):
                if sem_augs[0] != sem_augs[i]:
                    keep_random = True

            while keep_random and sem_aug == items:
                sem_aug = random.choice(sem_augs)

            sem_aug = sem_aug[:-1]
            pad_len = self.max_len - len(sem_aug)
            sem_aug = [0] * pad_len + sem_aug
            sem_aug = sem_aug[-self.max_len:]
            assert len(sem_aug) == self.max_len

            cur_tensors = (
                torch.tensor(self.user_ids[index], dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(neg_answer, dtype=torch.long),
                torch.tensor(sem_aug, dtype=torch.long),
                torch.tensor(time_input, dtype=torch.long), # time span ids
            )

        else:
            cur_tensors = (
                torch.tensor(self.user_ids[index], dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(neg_answer, dtype=torch.long),
                torch.zeros(0, dtype=torch.long), # not used
                torch.tensor(time_input, dtype=torch.long), # time span ids
            )

        return cur_tensors


def neg_sample(item_set, item_size):
    item = random.randint(1, item_size - 1)
    while item in item_set:
        item = random.randint(1, item_size - 1)
    return item

def generate_rating_matrix_valid(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-2]: #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix

def generate_rating_matrix_test(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-1]: #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix

def get_rating_matrix(data_name, seq_dic, max_item):
    
    num_items = max_item + 1
    valid_rating_matrix = generate_rating_matrix_valid(seq_dic['user_seq'], seq_dic['num_users'], num_items)
    test_rating_matrix = generate_rating_matrix_test(seq_dic['user_seq'], seq_dic['num_users'], num_items)

    return valid_rating_matrix, test_rating_matrix

def get_user_seqs_and_max_item(data_file):
    lines = open(data_file).readlines()
    lines = lines[1:]
    user_seq = []
    item_set = set()
    for line in lines:
        user, items = line.strip().split('	', 1)
        items = items.split()
        items = [int(item) for item in items]
        user_seq.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)
    return user_seq, max_item

def get_user_seqs(data_file):
    lines = open(data_file).readlines()
    user_seq = []
    item_set = set()
    for line in lines:
        user, items = line.strip().split(' ', 1)
        items = items.split(' ')
        items = [int(item) for item in items]
        user_seq.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)
    num_users = len(lines)

    return user_seq, max_item, num_users

def get_seq_dic(args):

    # Always keep the original item sequence as the source of truth.
    # Time files are optional side features and must align with <data_name>.txt.
    args.data_file = args.data_dir + args.data_name + '.txt'
    user_seq, max_item, num_users = get_user_seqs(args.data_file)

    # Load time intervals if available
    time_file = args.data_dir + args.data_name + '_time_time.txt'
    if os.path.exists(time_file):
        # Check if pre-bucketed (TASIF-style absolute time spans)
        prebucketed = hasattr(args, 'time_thresholds') and args.time_thresholds == 'preset'
        if prebucketed:
            # Values are already discrete IDs, compute num_time_span_ids from data
            time_seq, _ = get_time_seqs(time_file, prebucketed=True)
            all_ids = [t for seq in time_seq for t in seq]
            args.num_time_span_ids = max(all_ids) + 2  # +1 for max id, +1 for padding
        else:
            if hasattr(args, 'time_thresholds') and args.time_thresholds:
                thresholds = [int(x) for x in args.time_thresholds.split(',')]
            else:
                thresholds = DEFAULT_TIME_THRESHOLDS
            time_seq, _ = get_time_seqs(time_file, thresholds)
            args.num_time_span_ids = len(thresholds) + 1
    else:
        time_seq = None

    if time_seq is not None:
        if len(time_seq) != len(user_seq):
            raise ValueError(
                f"Time sequence count mismatch for {args.data_name}: "
                f"{len(time_seq)} time rows vs {len(user_seq)} item rows. "
                f"Regenerate {time_file} from the same source as {args.data_file}."
            )
        for idx, (items, times) in enumerate(zip(user_seq, time_seq), start=1):
            if len(items) != len(times):
                raise ValueError(
                    f"Time sequence length mismatch for {args.data_name} user line {idx}: "
                    f"{len(times)} time ids vs {len(items)} items. "
                    f"Regenerate {time_file} from the same source as {args.data_file}."
                )

    seq_dic = {'user_seq': user_seq, 'time_seq': time_seq, 'num_users': num_users}

    return seq_dic, max_item, num_users

def get_dataloder(args,seq_dic):

    train_dataset = RecDataset(args, seq_dic['user_seq'], seq_dic.get('time_seq'), data_type='train')
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, num_workers=args.num_workers)

    eval_dataset = RecDataset(args, seq_dic['user_seq'], seq_dic.get('time_seq'), data_type='valid')
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size, num_workers=args.num_workers)

    test_dataset = RecDataset(args, seq_dic['user_seq'], seq_dic.get('time_seq'), data_type='test')
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size, num_workers=args.num_workers)

    return train_dataloader, eval_dataloader, test_dataloader
