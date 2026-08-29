import os
import random
import torch
import datetime
import argparse
import numpy as np
import logging

def set_logger(log_path, log_name='seqrec', mode='a'):
    """set up log file
    mode : 'a'/'w' mean append/overwrite,
    """
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_path, mode=mode)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.propagate = False
    return logger


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'{path} created')

def get_local_time():
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y_%H-%M-%S')

    return cur

def parse_args():

    parser = argparse.ArgumentParser()

    # basic args
    parser.add_argument("--data_dir", default="./data/", type=str)
    parser.add_argument("--output_dir", default="output/", type=str)
    parser.add_argument("--data_name", default="LastFM", type=str)
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--load_model", default=None, type=str)
    parser.add_argument(
        "--eval_split",
        choices=("valid", "test"),
        default="test",
        help="checkpoint split evaluated by --do_eval",
    )
    parser.add_argument("--train_name", default=get_local_time(), type=str)
    parser.add_argument("--num_items", default=10, type=int)
    parser.add_argument("--num_users", default=10, type=int)

    # train args
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate of adam")
    parser.add_argument("--batch_size", default=256, type=int, help="number of batch_size")
    parser.add_argument("--epochs", default=200, type=int, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", default=1, type=int, help="per epoch print res")
    parser.add_argument("--patience", default=10, type=int, help="epochs to wait after the development score stops improving")
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument(
        "--strict_full_sort_mask",
        action="store_true",
        help="exclude padding item 0 and seen items with -inf during full-sort evaluation",
    )
    parser.add_argument(
        "--skip_test_after_training",
        action="store_true",
        help=(
            "finish checkpoint selection without evaluating the test split"
        ),
    )

    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--weight_decay", default=0.0, type=float, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", default=0.9, type=float, help="adam first beta value")
    parser.add_argument("--adam_beta2", default=0.999, type=float, help="adam second beta value")
    parser.add_argument("--gpu_id", default="0", type=str, help="gpu_id")
    parser.add_argument("--variance", default=5, type=float)
    parser.add_argument(
        "--paired_gate_experiment",
        action="store_true",
        help=(
            "align gated models with the corresponding TCF initialization "
            "and reset sampling RNG after model construction"
        ),
    )
    parser.add_argument(
        "--paired_time_factorial",
        action="store_true",
        help=(
            "initialize TCF-Time from the same TCF parameters and "
            "post-construction RNG state used by its paired TCF run"
        ),
    )
    parser.add_argument(
        "--log_gate_stats",
        action="store_true",
        help="log per-layer/head gate statistics by visible history length",
    )
    parser.add_argument(
        "--gate_residual_scale",
        default=0.10,
        type=float,
        help="maximum per-sample residual logit shift around the static gate",
    )
    parser.add_argument(
        "--short_branch_kernel_size",
        default=3,
        type=int,
        help="odd Conv1d kernel size for the short-history FE-GRU branch",
    )
    parser.add_argument(
        "--short_branch_init_weight",
        default=0.0,
        type=float,
        help=(
            "initial FE-GRU branch weight for short-history models; 0.0 "
            "keeps paired initialization identical to TCF-Time"
        ),
    )
    parser.add_argument(
        "--short_history_threshold",
        default=5,
        type=int,
        help="enable the short-history branch when visible history length <= this value",
    )
    parser.add_argument(
        "--short_history_mode",
        default="hard",
        type=str,
        choices=["hard", "soft"],
        help="hard threshold or soft sigmoid gate for short-history branch usage",
    )
    parser.add_argument(
        "--short_history_temperature",
        default=2.0,
        type=float,
        help="temperature for --short_history_mode soft",
    )
    parser.add_argument(
        "--short_branch_layer_scope",
        default="first",
        type=str,
        choices=["first", "all"],
        help="which TCF-Time layers receive the short-history branch",
    )

    # model args
    parser.add_argument("--model_type", default='TCF', type=str)
    parser.add_argument("--max_seq_length", default=50, type=int)
    parser.add_argument("--hidden_size", default=64, type=int, help="embedding dimension")
    parser.add_argument("--num_hidden_layers", default=2, type=int, help="number of blocks") ##################################### 16 ##############################################
    parser.add_argument("--hidden_act", default="gelu", type=str) # gelu relu
    parser.add_argument("--num_attention_heads", default=2, type=int)
    parser.add_argument("--attention_probs_dropout_prob", default=0.5, type=float)
    parser.add_argument("--hidden_dropout_prob", default=0.5, type=float)
    parser.add_argument("--initializer_range", default=0.02, type=float)

    args, _ = parser.parse_known_args()

    if args.model_type.lower() == 'bsarec':
        parser.add_argument("--c", default=3, type=int)
        parser.add_argument("--alpha", default=0.9, type=float)

    elif args.model_type.lower() == 'bert4rec':
        parser.add_argument("--mask_ratio", default=0.2, type=float)

    elif args.model_type.lower() == 'caser':
        parser.add_argument("--nh", default=8, type=int)
        parser.add_argument("--nv", default=4, type=int)
        parser.add_argument("--reg_weight", default=1e-4, type=float)

    elif args.model_type.lower() == 'duorec':
        parser.add_argument("--tau", default=1.0, type=float)
        parser.add_argument("--lmd", default=0.1, type=float)
        parser.add_argument("--lmd_sem", default=0.1, type=float)
        parser.add_argument("--ssl", default='us_x', type=str)
        parser.add_argument("--sim", default='dot', type=str)

    elif args.model_type.lower() == 'fearec':
        parser.add_argument("--tau", default=1.0, type=float)
        parser.add_argument("--lmd", default=0.1, type=float)
        parser.add_argument("--lmd_sem", default=0.1, type=float)
        parser.add_argument("--ssl", default='us_x', type=str)
        parser.add_argument("--sim", default='dot', type=str)
        parser.add_argument("--spatial_ratio", default=0.1, type=float)
        parser.add_argument("--global_ratio", default=0.6, type=float)
        parser.add_argument("--fredom_type", default='us_x', type=str)
        parser.add_argument("--fredom", default='True', type=str) # use eval function to use as boolean

    elif args.model_type.lower() == 'slime4rec':
        parser.add_argument("--tau", default=1.0, type=float)
        parser.add_argument("--lmd", default=0.1, type=float)
        parser.add_argument("--lmd_sem", default=0.1, type=float)
        parser.add_argument("--ssl", default='us_x', type=str)
        parser.add_argument("--sim", default='dot', type=str)
        parser.add_argument("--spatial_ratio", default=0.1, type=float)
        parser.add_argument("--filter_mixer", default='M', type=str)
        parser.add_argument("--dynamic_ratio", default=0.8, type=float)
        parser.add_argument("--slide_mode", default='four', type=str)

    elif args.model_type.lower() == 'gru4rec':
        parser.add_argument("--gru_hidden_size", default=64, type=int, help="hidden size of GRU")

    elif args.model_type.lower() == "tisasrec":
        parser.add_argument(
            "--num_time_span_ids",
            default=14,
            type=int,
            help="number of aligned temporal IDs loaded by the dataset",
        )
        parser.add_argument(
            "--time_thresholds",
            default=None,
            type=str,
            help="use 'preset' to consume aligned temporal IDs without rebucketing",
        )
        parser.add_argument(
            "--tisasrec_time_span",
            default=12,
            type=int,
            help="maximum clipped pairwise temporal-ID distance",
        )

    elif args.model_type.lower() in {
        "wearec",
        "tcf",
        "tcf_markov",
        "tcf_static_gate",
        "tcf_selective_gate",
        "tcf_residual_gate",
    }:
        parser.add_argument("--num_heads", default=2, type=int)
        parser.add_argument("--alpha", default=0.3, type=float)
        if args.model_type.lower() == "tcf":
            parser.add_argument(
                "--num_time_span_ids",
                default=14,
                type=int,
                help="number of time interval buckets",
            )
            parser.add_argument(
                "--time_thresholds",
                default=None,
                type=str,
                help="use 'preset' to consume aligned temporal IDs",
            )
            parser.add_argument(
                "--time_embedding_scale",
                default=1.0,
                type=float,
                help="scale applied to time embeddings",
            )
        if args.model_type.lower() == "tcf_markov":
            parser.add_argument(
                "--markov_lambda",
                default=0.0,
                type=float,
                help="weight for the non-learned Markov transition score",
            )
            parser.add_argument(
                "--markov_transition_window",
                default=3,
                type=int,
                help="number of recent visible items used for Markov scoring",
            )
            parser.add_argument(
                "--markov_transition_alpha",
                default=2.0,
                type=float,
                help="distance decay used when building window-mode transitions",
            )
            parser.add_argument(
                "--markov_recency_tau",
                default=1.5,
                type=float,
                help="exponential recency decay over recent source items",
            )
            parser.add_argument(
                "--markov_transition_mode",
                default="last",
                type=str,
                choices=["last", "window"],
                help="build adjacent-item or window co-occurrence transition table",
            )

    elif args.model_type.lower() in {
        "tcf_time",
        "tcf_time_static_gate",
        "tcf_time_selective_gate",
        "tcf_time_residual_gate",
        "tcf_time_short_history",
        "tcf_time_short_intent",
        "tcf_time_short_memory",
        "tcf_time_post_short_memory",
        "tcf_time_short_score",
        "tcf_time_transition",
        "tcf_time_markov",
        "tcf_time_wavelet_markov",
        "tcf_time_wavelet_basis_markov",
        "tcf_time_tg",
    }:
        parser.add_argument("--num_heads", default=2, type=int)
        parser.add_argument("--alpha", default=0.3, type=float)
        parser.add_argument("--num_time_span_ids", default=14, type=int, help="number of time interval buckets")
        parser.add_argument("--time_thresholds", default=None, type=str, help="comma-separated time thresholds, e.g. '0' for binary")
        parser.add_argument("--time_embedding_scale", default=1.0, type=float, help="scale applied to time embeddings before adding item embeddings")
        if args.model_type.lower() == "tcf_time_short_intent":
            parser.add_argument(
                "--short_intent_threshold",
                default=5,
                type=int,
                help="history length where the short-intent soft gate is centered",
            )
            parser.add_argument(
                "--short_intent_temperature",
                default=2.0,
                type=float,
                help="temperature for the short-intent length gate",
            )
            parser.add_argument(
                "--short_intent_min_gate",
                default=0.0,
                type=float,
                help="minimum non-empty short-intent gate for longer histories",
            )
            parser.add_argument(
                "--short_intent_window",
                default=5,
                type=int,
                help="number of most recent visible items used by recency pooling",
            )
            parser.add_argument(
                "--short_intent_recency_tau",
                default=2.0,
                type=float,
                help="exponential decay factor for recent-item pooling",
            )
            parser.add_argument(
                "--short_intent_init_weight",
                default=0.0,
                type=float,
                help="initial bounded mix weight for the short-intent expert",
            )
            parser.add_argument(
                "--short_intent_max_weight",
                default=0.20,
                type=float,
                help="maximum absolute short-intent mix weight",
            )
            parser.add_argument(
                "--short_intent_gate_bias",
                default=2.0,
                type=float,
                help="initial bias for the learned short-intent gate",
            )
            parser.add_argument(
                "--short_intent_mix_position",
                default="last",
                type=str,
                choices=["last", "all"],
                help="inject the short expert only at the last position or all visible positions",
            )
            parser.add_argument(
                "--short_intent_layer_scope",
                default="first",
                type=str,
                choices=["first", "all"],
                help="which TCF-Time layers receive the short-intent expert",
            )
        if args.model_type.lower() == "tcf_time_short_memory":
            parser.add_argument(
                "--short_memory_threshold",
                default=5,
                type=int,
                help="history length where the short-memory soft gate is centered",
            )
            parser.add_argument(
                "--short_memory_temperature",
                default=2.0,
                type=float,
                help="temperature for the short-memory length gate",
            )
            parser.add_argument(
                "--short_memory_min_gate",
                default=0.0,
                type=float,
                help="minimum non-empty short-memory gate for longer histories",
            )
            parser.add_argument(
                "--short_memory_window",
                default=5,
                type=int,
                help="number of recent visible items used by short-query pooling",
            )
            parser.add_argument(
                "--short_memory_recency_tau",
                default=2.0,
                type=float,
                help="exponential decay factor for recent-item pooling",
            )
            parser.add_argument(
                "--short_memory_time_tau",
                default=4.0,
                type=float,
                help="time-bucket decay factor for short-query pooling and memory bias",
            )
            parser.add_argument(
                "--short_memory_init_weight",
                default=0.0,
                type=float,
                help="initial bounded mix weight for the short-memory reader",
            )
            parser.add_argument(
                "--short_memory_max_weight",
                default=0.20,
                type=float,
                help="maximum absolute short-memory mix weight",
            )
            parser.add_argument(
                "--short_memory_gate_bias",
                default=2.0,
                type=float,
                help="initial bias for the learned short-memory mix gate",
            )
            parser.add_argument(
                "--short_memory_mix_position",
                default="last",
                type=str,
                choices=["last", "all"],
                help="inject the memory readout only at the last position or all visible positions",
            )
            parser.add_argument(
                "--short_memory_layer_scope",
                default="first",
                type=str,
                choices=["first", "all"],
                help="which TCF-Time layers receive the short-memory reader",
            )
        if args.model_type.lower() == "tcf_time_post_short_memory":
            parser.add_argument(
                "--post_short_threshold",
                default=5,
                type=int,
                help="history length where the post-encoder short-memory gate is centered",
            )
            parser.add_argument(
                "--post_short_gate_mode",
                default="soft",
                type=str,
                choices=["hard", "soft", "none"],
                help="hard short-only gate, soft sigmoid length gate, or no length gate",
            )
            parser.add_argument(
                "--post_short_temperature",
                default=2.0,
                type=float,
                help="temperature for --post_short_gate_mode soft",
            )
            parser.add_argument(
                "--post_short_min_gate",
                default=0.0,
                type=float,
                help="minimum non-empty gate for longer histories in soft mode",
            )
            parser.add_argument(
                "--post_short_window",
                default=5,
                type=int,
                help="number of recent encoded states used by post short intent",
            )
            parser.add_argument(
                "--post_short_recency_tau",
                default=2.0,
                type=float,
                help="exponential recency decay for post short intent",
            )
            parser.add_argument(
                "--post_short_time_tau",
                default=4.0,
                type=float,
                help="time-bucket decay for post short intent and memory bias",
            )
            parser.add_argument(
                "--post_short_hops",
                default=1,
                type=int,
                help="number of short-query memory reading hops",
            )
            parser.add_argument(
                "--post_short_kernel_size",
                default=3,
                type=int,
                help="odd Conv1d kernel size for post short feature extraction",
            )
            parser.add_argument(
                "--post_short_init_weight",
                default=0.0,
                type=float,
                help="initial bounded post short-memory mix weight; 0.0 keeps exact TCF-Time",
            )
            parser.add_argument(
                "--post_short_max_weight",
                default=0.20,
                type=float,
                help="maximum absolute post short-memory mix weight",
            )
            parser.add_argument(
                "--post_short_gate_bias",
                default=2.0,
                type=float,
                help="initial bias for the learned post short-memory mix gate",
            )
        if args.model_type.lower() == "tcf_time_short_score":
            parser.add_argument(
                "--short_score_threshold",
                default=5,
                type=int,
                help="maximum visible history length that activates the short-score expert",
            )
            parser.add_argument(
                "--short_score_gate_mode",
                default="hard",
                type=str,
                choices=["hard", "soft"],
                help="hard short-only gate or soft sigmoid length gate",
            )
            parser.add_argument(
                "--short_score_temperature",
                default=2.0,
                type=float,
                help="temperature for --short_score_gate_mode soft",
            )
            parser.add_argument(
                "--short_score_window",
                default=5,
                type=int,
                help="number of recent visible hidden states used by the score expert",
            )
            parser.add_argument(
                "--short_score_recency_tau",
                default=2.0,
                type=float,
                help="exponential recency decay for the score expert",
            )
            parser.add_argument(
                "--short_score_time_tau",
                default=4.0,
                type=float,
                help="time-bucket decay for the score expert",
            )
            parser.add_argument(
                "--short_score_init_weight",
                default=0.0,
                type=float,
                help="initial bounded score-expert weight; 0.0 keeps exact TCF-Time logits",
            )
            parser.add_argument(
                "--short_score_max_weight",
                default=0.20,
                type=float,
                help="maximum absolute score-expert weight",
            )
        if args.model_type.lower() == "tcf_time_transition":
            parser.add_argument(
                "--transition_dim",
                default=64,
                type=int,
                help="rank of the factorized item-to-item transition expert",
            )
            parser.add_argument(
                "--transition_window",
                default=3,
                type=int,
                help="number of recent visible items used by the transition expert",
            )
            parser.add_argument(
                "--transition_recency_tau",
                default=1.5,
                type=float,
                help="exponential recency decay for recent transition sources",
            )
            parser.add_argument(
                "--transition_time_tau",
                default=4.0,
                type=float,
                help="time-bucket decay for recent transition sources",
            )
            parser.add_argument(
                "--transition_gate_mode",
                default="none",
                type=str,
                choices=["none", "hard", "soft"],
                help="optional length gate for the transition expert",
            )
            parser.add_argument(
                "--transition_threshold",
                default=50,
                type=int,
                help="history length threshold for hard/soft transition gating",
            )
            parser.add_argument(
                "--transition_temperature",
                default=2.0,
                type=float,
                help="temperature for --transition_gate_mode soft",
            )
            parser.add_argument(
                "--transition_min_gate",
                default=0.0,
                type=float,
                help="minimum non-empty transition length gate in soft mode",
            )
            parser.add_argument(
                "--transition_init_weight",
                default=0.0,
                type=float,
                help="initial bounded transition weight; 0.0 keeps TCF-Time logits",
            )
            parser.add_argument(
                "--transition_max_weight",
                default=1.0,
                type=float,
                help="maximum absolute transition expert weight",
            )
            parser.add_argument(
                "--transition_gate_bias",
                default=2.0,
                type=float,
                help="initial bias for the learned transition gate",
            )
        if args.model_type.lower() in {
            "tcf_time_markov",
            "tcf_time_wavelet_markov",
            "tcf_time_wavelet_basis_markov",
        }:
            parser.add_argument(
                "--markov_lambda",
                default=0.0,
                type=float,
                help="weight for the non-learned Markov transition score",
            )
            parser.add_argument(
                "--markov_transition_window",
                default=3,
                type=int,
                help="number of recent visible items used for Markov scoring",
            )
            parser.add_argument(
                "--markov_transition_alpha",
                default=2.0,
                type=float,
                help="distance decay used when building window-mode transitions",
            )
            parser.add_argument(
                "--markov_recency_tau",
                default=1.5,
                type=float,
                help="exponential recency decay over recent source items",
            )
            parser.add_argument(
                "--markov_transition_mode",
                default="last",
                type=str,
                choices=["last", "window"],
                help="build adjacent-item or window co-occurrence transition table",
            )
        if args.model_type.lower() in {
            "tcf_time_wavelet_markov",
            "tcf_time_wavelet_basis_markov",
        }:
            parser.add_argument(
                "--filter_type",
                default="haar",
                type=str,
                choices=["haar", "db2", "coiflet", "meyer", "morlet", "mexican_hat"],
                help="local wavelet filter/basis type",
            )
            parser.add_argument(
                "--filter_length",
                default=8,
                type=int,
                help="filter length for meyer/morlet/mexican_hat filters",
            )
            parser.add_argument(
                "--sigma",
                default=1.0,
                type=float,
                help="sigma for morlet/mexican_hat filters",
            )
        if args.model_type.lower() == "tcf_time_wavelet_markov":
            parser.add_argument(
                "--pass_weight",
                default=0.7,
                type=float,
                help="scaling factor applied to generated wavelet filters",
            )
            parser.add_argument(
                "--wavelet_beta_init_scale",
                default=1.0,
                type=float,
                help="initial std scale for the high-pass beta parameter",
            )
            parser.add_argument(
                "--wavelet_detach_conv",
                action="store_true",
                help="match WaveRec's detached convolution outputs",
            )
        if args.model_type.lower() == "tcf_time_tg":
            parser.add_argument(
                "--tg_layer_scale_init",
                default=1e-3,
                type=float,
                help="initial residual scale for TG-TCF branch fusion",
            )
            parser.add_argument(
                "--tg_residual_prior",
                default=0.10,
                type=float,
                help="initial softmax prior mass for the residual expert",
            )
            parser.add_argument(
                "--tg_detail_gate_bias",
                default=2.0,
                type=float,
                help="initial bias before sigmoid for the wavelet detail gate",
            )
            parser.add_argument(
                "--tg_gate_temperature",
                default=1.0,
                type=float,
                help="softmax temperature for the three-branch TG gate",
            )
            parser.add_argument(
                "--tg_gate_hidden_ratio",
                default=1.0,
                type=float,
                help="hidden-size multiplier for TG gate MLPs",
            )
            parser.add_argument(
                "--tg_gate_dropout",
                default=0.0,
                type=float,
                help="dropout inside TG gate MLPs",
            )

    return parser.parse_args()

class EarlyStopping:
    """Stop training when the held-out development score stops improving."""
    def __init__(self, checkpoint_path, logger, patience=10, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after the development score improves.
                            Default: 10
            verbose (bool): If True, logs each development-score improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.logger = logger

    def compare(self, score):
        for i in range(len(score)):
            if score[i] > self.best_score[i]+self.delta:
                return False
        return True

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.score_min = np.array([0]*len(score))
            self.save_checkpoint(score, model)
        elif self.compare(score):
            self.counter += 1
            self.logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        '''Save the model when the development score improves.'''
        if self.verbose:
            self.logger.info(f'Development score increased.  Saving model ...')
        torch.save(model.state_dict(), self.checkpoint_path)
        self.score_min = score
