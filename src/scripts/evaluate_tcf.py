#!/usr/bin/env python3
"""Strict full-sort evaluation for a freshly trained TCF checkpoint."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from dataset import get_dataloder, get_rating_matrix, get_seq_dic
from model.tcf import (
    TCFModel,
    build_markov_transition_index,
)


METRICS = ("HR@10", "NDCG@10", "HR@20", "NDCG@20")


def model_args(
    preset: dict[str, Any],
    batch_size: int,
    *,
    data_dir: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        data_dir=data_dir,
        output_dir="",
        data_name=str(preset["data_name"]),
        do_eval=True,
        load_model=None,
        eval_split="test",
        train_name="tcf_evaluation",
        num_items=10,
        num_users=10,
        lr=float(preset["lr"]),
        batch_size=batch_size,
        epochs=200,
        no_cuda=False,
        log_freq=1,
        patience=10,
        num_workers=0,
        strict_full_sort_mask=True,
        skip_test_after_training=True,
        seed=int(preset["seed"]),
        weight_decay=float(preset["weight_decay"]),
        adam_beta1=0.9,
        adam_beta2=0.999,
        gpu_id="0",
        variance=5,
        paired_gate_experiment=False,
        log_gate_stats=False,
        gate_residual_scale=0.1,
        short_branch_kernel_size=3,
        short_branch_init_weight=0.0,
        short_history_threshold=5,
        short_history_mode="hard",
        short_history_temperature=2.0,
        short_branch_layer_scope="first",
        model_type="tcf",
        max_seq_length=50,
        hidden_size=64,
        num_hidden_layers=2,
        hidden_act="gelu",
        num_attention_heads=2,
        attention_probs_dropout_prob=float(preset["dropout"]),
        hidden_dropout_prob=float(preset["dropout"]),
        initializer_range=0.02,
        num_heads=int(preset["num_heads"]),
        alpha=float(preset["alpha"]),
        num_time_span_ids=14,
        time_thresholds="preset",
        time_embedding_scale=float(preset["time_embedding_scale"]),
        markov_lambda=1.0,
        markov_transition_window=3,
        markov_transition_alpha=2.0,
        markov_recency_tau=1.5,
        markov_transition_mode="last",
        complex_dff_epsilon=0.0,
        complex_dff_rank=0,
        complex_dff_log_amp_limit=2.0,
        complex_dff_phase_limit=math.pi,
        complex_dff_disable_amplitude=False,
        complex_dff_disable_phase=False,
        wavelet_basis=str(preset.get("wavelet_basis", "haar")),
        paired_time_factorial=False,
    )


@dataclass
class DatasetContext:
    signature: tuple[Any, ...]
    item_size: int
    num_users: int
    num_time_span_ids: int
    test_loader: Any
    test_matrix: Any
    transition_scores: dict[int, tuple[list[int], list[float]]]
    sparse_cache: dict[tuple[str, torch.dtype, int], torch.Tensor]
    transition_stats: dict[str, int]


def evaluation_args(job: Any, batch_size: int) -> SimpleNamespace:
    data_dir = str(Path(job.preset["data_dir"]).resolve()) + "/"
    args = model_args(job.preset, batch_size, data_dir=data_dir)
    args.complex_dff_epsilon = float(job.epsilon)
    args.complex_dff_rank = int(job.rank)
    args.complex_dff_disable_amplitude = bool(
        job.preset.get("complex_dff_disable_amplitude", False)
    )
    args.complex_dff_disable_phase = bool(
        job.preset.get("complex_dff_disable_phase", False)
    )
    return args


def context_signature(args: SimpleNamespace, batch_size: int) -> tuple[Any, ...]:
    return (
        str(Path(args.data_dir).resolve()),
        str(args.data_name),
        int(args.max_seq_length),
        int(batch_size),
        str(args.markov_transition_mode),
        float(args.markov_transition_alpha),
    )


def build_context(job: Any, batch_size: int) -> DatasetContext:
    args = evaluation_args(job, batch_size)
    seq_dic, max_item, num_users = get_seq_dic(args)
    args.item_size = max_item + 1
    args.num_users = num_users + 1
    args.same_target_path = str(
        Path(args.data_dir) / f"{args.data_name}_same_target.npy"
    )
    _, _, test_loader = get_dataloder(args, seq_dic)
    _, test_matrix = get_rating_matrix(args.data_name, seq_dic, max_item)
    transition_scores = build_markov_transition_index(
        seq_dic["user_seq"],
        args.max_seq_length,
        args.markov_transition_mode,
        args.markov_transition_alpha,
    )
    return DatasetContext(
        signature=context_signature(args, batch_size),
        item_size=args.item_size,
        num_users=args.num_users,
        num_time_span_ids=args.num_time_span_ids,
        test_loader=test_loader,
        test_matrix=test_matrix,
        transition_scores=transition_scores,
        sparse_cache={},
        transition_stats={
            "source_items": len(transition_scores),
            "transitions": sum(len(targets) for targets, _ in transition_scores.values()),
        },
    )


@torch.no_grad()
def evaluate_test_grid(
    job: Any,
    *,
    windows: tuple[int, ...],
    taus: tuple[float, ...],
    lambdas: tuple[float, ...],
    eval_batch_size: int,
    device: torch.device,
    dataset_context: DatasetContext | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    args = evaluation_args(job, eval_batch_size)
    context = dataset_context or build_context(job, eval_batch_size)
    if context.signature != context_signature(args, eval_batch_size):
        raise ValueError("dataset context does not match the evaluation job")
    args.item_size = context.item_size
    args.num_users = context.num_users
    args.num_time_span_ids = context.num_time_span_ids

    model = TCFModel(args)
    model._markov_transition_scores = context.transition_scores
    model._markov_sparse_cache = context.sparse_cache
    state = torch.load(job.checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    totals_by_setting: dict[tuple[int, float, float], np.ndarray] = defaultdict(
        lambda: np.zeros(4, dtype=np.float64)
    )
    user_count = 0
    for batch in context.test_loader:
        user_ids, input_ids, answers, _, _, ts_seq = (
            tensor.to(device) for tensor in batch
        )
        sequence_output = model.forward(
            input_ids,
            ts_seq=ts_seq,
            user_ids=user_ids,
        )
        base_logits = torch.matmul(
            sequence_output[:, -1, :],
            model.item_embeddings.weight.transpose(0, 1),
        )
        users = user_ids.detach().cpu().numpy()
        seen = torch.as_tensor(
            context.test_matrix[users].toarray() > 0,
            dtype=torch.bool,
            device=device,
        )
        targets = answers.reshape(-1, 1)
        user_count += answers.numel()

        for window in windows:
            model.markov_transition_window = window
            for tau in taus:
                model.markov_recency_tau = tau
                model.markov_lambda = 1.0
                transition_logits = model.markov_transition_logits(
                    input_ids,
                    args.item_size,
                )
                for lambda_value in lambdas:
                    scores = base_logits + transition_logits * lambda_value
                    scores.masked_fill_(seen, float("-inf"))
                    scores[:, 0] = float("-inf")
                    prediction = torch.topk(
                        scores,
                        k=min(20, scores.shape[-1]),
                        dim=-1,
                        largest=True,
                        sorted=True,
                    ).indices
                    matches = prediction.eq(targets)
                    has_hit = matches.any(dim=-1)
                    ranks = matches.to(torch.long).argmax(dim=-1)
                    hit10 = has_hit & ranks.lt(10)
                    hit20 = has_hit & ranks.lt(20)
                    discount = torch.reciprocal(
                        torch.log2(ranks.to(torch.float64) + 2.0)
                    )
                    totals_by_setting[(window, tau, lambda_value)] += np.asarray(
                        [
                            hit10.sum(dtype=torch.float64).item(),
                            (discount * hit10).sum().item(),
                            hit20.sum(dtype=torch.float64).item(),
                            (discount * hit20).sum().item(),
                        ],
                        dtype=np.float64,
                    )

    if user_count == 0:
        raise ValueError(f"empty test loader: {job.dataset_key}")
    rows = []
    for window in windows:
        for tau in taus:
            for lambda_value in lambdas:
                totals = totals_by_setting[(window, tau, lambda_value)]
                rows.append(
                    {
                        "dataset": job.dataset_key,
                        "candidate": job.candidate,
                        "epsilon": job.epsilon,
                        "rank": job.rank,
                        "seed": int(args.seed),
                        "window": window,
                        "tau": tau,
                        "lambda": lambda_value,
                        **{
                            metric: float(totals[index] / user_count)
                            for index, metric in enumerate(METRICS)
                        },
                        "strict_full_sort_mask": True,
                        "checkpoint": str(job.checkpoint),
                    }
                )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows, context.transition_stats
