import tqdm
import torch
import numpy as np

from torch.optim import Adam
from metrics import recall_at_k, ndcg_k


def mask_full_sort_scores(
    rating_pred,
    train_matrix,
    user_indices,
    strict=False,
):
    """Mask seen items consistently for neural and non-neural full-sort runs."""

    seen = train_matrix[user_indices].toarray() > 0
    if rating_pred.shape[1] != seen.shape[1]:
        rating_pred = rating_pred[:, : seen.shape[1]]
    fill_value = -np.inf if strict else 0.0
    rating_pred[seen] = fill_value
    if strict and rating_pred.shape[1] > 0:
        rating_pred[:, 0] = -np.inf
    return rating_pred

class Trainer:
    def __init__(self, model, train_dataloader, eval_dataloader, test_dataloader, args, logger):
        super(Trainer, self).__init__()

        self.args = args
        self.logger = logger
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        self.model = model
        if self.cuda_condition:
            self.model.cuda()

        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        # self.data_name = self.args.data_name
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(self.model.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)

        self.logger.info(f"Total Parameters: {sum([p.nelement() for p in self.model.parameters()])}")
        self.use_time = self.args.model_type.lower() in {
            "tcf",
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
            "tisasrec",
        }
        self.log_gate_stats = (
            getattr(self.args, "log_gate_stats", False)
            and hasattr(self.model, "get_last_gate_weights")
        )

    def update_gate_stats(self, accumulator, input_ids):
        gate_weights = self.model.get_last_gate_weights()
        if not gate_weights or any(weights is None for weights in gate_weights):
            return accumulator

        if accumulator is None:
            accumulator = []
            for weights in gate_weights:
                layer_stats = {}
                for group_name in ["empty", "short_1_5", "medium_6_10", "long_gt_10"]:
                    layer_stats[group_name] = {
                        "count": 0,
                        "sum": torch.zeros(
                            weights.size(1), 2, dtype=torch.float64
                        ),
                        "entropy_sum": torch.zeros(
                            weights.size(1), dtype=torch.float64
                        ),
                        "wavelet_values": [],
                    }
                accumulator.append(layer_stats)

        history_length = input_ids.ne(0).sum(dim=1)
        group_masks = {
            "empty": history_length.eq(0),
            "short_1_5": history_length.ge(1) & history_length.le(5),
            "medium_6_10": history_length.ge(6) & history_length.le(10),
            "long_gt_10": history_length.gt(10),
        }

        for layer_index, weights in enumerate(gate_weights):
            detached = weights.detach()
            entropy = -(detached * detached.clamp_min(1e-12).log()).sum(dim=-1)
            for group_name, group_mask in group_masks.items():
                if not group_mask.any():
                    continue
                selected_weights = detached[group_mask]
                selected_entropy = entropy[group_mask]
                stats = accumulator[layer_index][group_name]
                stats["count"] += selected_weights.size(0)
                stats["sum"] += selected_weights.sum(dim=0).double().cpu()
                stats["entropy_sum"] += (
                    selected_entropy.sum(dim=0).double().cpu()
                )
                stats["wavelet_values"].append(
                    selected_weights[..., 0].float().cpu()
                )

        return accumulator

    def write_gate_stats(self, accumulator):
        if accumulator is None:
            return

        summary = {}
        for layer_index, layer_stats in enumerate(accumulator):
            for group_name, stats in layer_stats.items():
                if stats["count"] == 0:
                    continue
                mean = stats["sum"] / stats["count"]
                entropy = stats["entropy_sum"] / stats["count"]
                wavelet_values = torch.cat(
                    stats["wavelet_values"], dim=0
                )
                wavelet_std = wavelet_values.std(dim=0, unbiased=False)
                wavelet_quantiles = torch.quantile(
                    wavelet_values,
                    torch.tensor([0.1, 0.5, 0.9]),
                    dim=0,
                )
                summary[f"layer_{layer_index}/{group_name}"] = {
                    "count": stats["count"],
                    "wavelet": [round(value, 6) for value in mean[:, 0].tolist()],
                    "fft": [round(value, 6) for value in mean[:, 1].tolist()],
                    "wavelet_std": [
                        round(value, 6) for value in wavelet_std.tolist()
                    ],
                    "wavelet_q10": [
                        round(value, 6)
                        for value in wavelet_quantiles[0].tolist()
                    ],
                    "wavelet_q50": [
                        round(value, 6)
                        for value in wavelet_quantiles[1].tolist()
                    ],
                    "wavelet_q90": [
                        round(value, 6)
                        for value in wavelet_quantiles[2].tolist()
                    ],
                    "entropy": [
                        round(value, 6) for value in entropy.tolist()
                    ],
                }
        self.logger.info({"GateStats": summary})

    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader, train=True)

    def valid(self, epoch):
        self.args.train_matrix = self.args.valid_rating_matrix
        return self.iteration(epoch, self.eval_dataloader, train=False)

    def test(self, epoch):
        self.args.train_matrix = self.args.test_rating_matrix
        return self.iteration(epoch, self.test_dataloader, train=False)

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        original_state_dict = self.model.state_dict()
        self.logger.info(original_state_dict.keys())
        new_dict = torch.load(file_name)
        self.logger.info(new_dict.keys())
        for key in new_dict:
            original_state_dict[key]=new_dict[key]
        self.model.load_state_dict(original_state_dict)

    def predict_full(self, seq_out):
        # [item_num hidden_size]
        test_item_emb = self.model.item_embeddings.weight
        # [batch hidden_size ]
        # import pdb; pdb.set_trace()
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred

    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [10, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "HR@10": '{:.8f}'.format(recall[0]),
            "NDCG@10": '{:.8f}'.format(ndcg[0]),
            "HR@20": '{:.8f}'.format(recall[1]),
            "NDCG@20": '{:.8f}'.format(ndcg[1]),
        }
        self.logger.info(post_fix)

        return [recall[0], ndcg[0], recall[1], ndcg[1]], str(post_fix)

    def iteration(self, epoch, dataloader, train=True):

        str_code = "train" if train else "test"
        # Setting the tqdm progress bar
        rec_data_iter = tqdm.tqdm(enumerate(dataloader),
                                  desc="Mode_%s:%d" % (str_code, epoch),
                                  total=len(dataloader),
                                  bar_format="{l_bar}{r_bar}")
        
        if train:
            self.model.train()
            rec_loss = 0.0

            for i, batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or CPU)
                batch = tuple(t.to(self.device) for t in batch)

                user_ids, input_ids, answers, neg_answer, same_target, ts_seq = batch
                if self.use_time:
                    loss = self.model.calculate_loss(input_ids, answers, neg_answer, same_target, user_ids, ts_seq)
                else:
                    loss = self.model.calculate_loss(input_ids, answers, neg_answer, same_target, user_ids)
                    
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                rec_loss += loss.item()

            post_fix = {
                "epoch": epoch,
                "rec_loss": '{:.4f}'.format(rec_loss / len(rec_data_iter)),
            }

            if (epoch + 1) % self.args.log_freq == 0:
                self.logger.info(str(post_fix))

        else:
            self.model.eval()
            pred_list = None
            answer_list = None
            gate_stats = None

            for i, batch in rec_data_iter:
                batch = tuple(t.to(self.device) for t in batch)
                user_ids, input_ids, answers, _, _, ts_seq = batch
                if hasattr(self.model, "full_sort_predict"):
                    if self.use_time:
                        rating_pred = self.model.full_sort_predict(
                            input_ids,
                            user_ids,
                            ts_seq,
                        )
                    else:
                        rating_pred = self.model.full_sort_predict(input_ids, user_ids)
                elif self.use_time:
                    recommend_output = self.model.predict(input_ids, user_ids, ts_seq)
                    if self.log_gate_stats:
                        gate_stats = self.update_gate_stats(gate_stats, input_ids)
                    recommend_output = recommend_output[:, -1, :]# 推荐的结果
                    rating_pred = self.predict_full(recommend_output)
                else:
                    recommend_output = self.model.predict(input_ids, user_ids)
                    if self.log_gate_stats:
                        gate_stats = self.update_gate_stats(gate_stats, input_ids)
                    recommend_output = recommend_output[:, -1, :]# 推荐的结果
                    rating_pred = self.predict_full(recommend_output)
                rating_pred = rating_pred.cpu().data.numpy().copy()
                batch_user_index = user_ids.cpu().numpy()
                
                rating_pred = mask_full_sort_scores(
                    rating_pred,
                    self.args.train_matrix,
                    batch_user_index,
                    strict=getattr(self.args, "strict_full_sort_mask", False),
                )

                # argpartition time complexity O(n)  argsort O(nlogn)
                # The minus sign "-" indicates a larger value.
                ind = np.argpartition(rating_pred, -20)[:, -20:]
                # Take the corresponding values from the corresponding dimension 
                # according to the returned subscript to get the sub-table of each row of topk
                arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                # Sort the sub-tables in order of magnitude.
                arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                # retrieve the original subscript from index again
                batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                if i == 0:
                    pred_list = batch_pred_list
                    answer_list = answers.cpu().data.numpy()
                else:
                    pred_list = np.append(pred_list, batch_pred_list, axis=0)
                    answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)

            if self.log_gate_stats:
                self.write_gate_stats(gate_stats)
            return self.get_full_sort_score(epoch, answer_list, pred_list)
