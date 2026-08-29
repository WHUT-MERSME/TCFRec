"""Final TCF model with temporal encoding and a complex dynamic frequency residual.

This submission-facing module contains the complete final TCF implementation:
the WEARec-derived encoder, temporal-state encoding, complex dynamic frequency
filter, and deterministic train-prefix calibration used during evaluation.

The original WEARec dynamic frequency filter learns real-valued scale and
bias terms for every head/frequency pair.  This module preserves that path and
adds the following complex residual on top of its filtered spectrum::

    G = exp(delta_log_amplitude) * exp(1j * delta_phase)
    F_out = F_legacy * (1 + epsilon * G)

The adaptive MLP is initialized to output zero, so amplitude=1 and phase=0.
Consequently, ``epsilon=0`` is exactly the legacy WEARec layer and the default
``epsilon=0.05`` starts close to it.  An optional frequency-specific low-rank
complex mixer introduces interactions across ``head_dim`` channels.  Its
output factor is initialized to zero and therefore does not perturb the
initial trajectory.
"""

from __future__ import annotations

import copy
import math
from collections import defaultdict

import torch
import torch.nn as nn

from model._abstract_model import SequentialRecModel
from model._modules import FeedForward, LayerNorm


SUPPORTED_WAVELET_BASES = (
    "haar",
    "db2",
    "coif1",
    "sym4",
    "bior4.4",
    "db6",
)

WAVELET_BASIS_METADATA = {
    "haar": {
        "canonical_name": "Haar",
        "family": "Haar",
        "filter_length_taps": 2,
        "orthogonal": True,
        "dual_analysis_synthesis": False,
    },
    "db2": {
        "canonical_name": "Db2",
        "family": "Daubechies",
        "filter_length_taps": 4,
        "orthogonal": True,
        "dual_analysis_synthesis": False,
    },
    "coif1": {
        "canonical_name": "Coif1",
        "family": "Coiflet",
        "filter_length_taps": 6,
        "orthogonal": True,
        "dual_analysis_synthesis": False,
    },
    "sym4": {
        "canonical_name": "Sym4",
        "family": "Symlet",
        "filter_length_taps": 8,
        "orthogonal": True,
        "dual_analysis_synthesis": False,
    },
    "bior4.4": {
        "canonical_name": "Bior4.4",
        "family": "Biorthogonal",
        "filter_length_taps": 10,
        "orthogonal": False,
        "dual_analysis_synthesis": True,
    },
    "db6": {
        "canonical_name": "Db6",
        "family": "Daubechies",
        "filter_length_taps": 12,
        "orthogonal": True,
        "dual_analysis_synthesis": False,
    },
}


def fixed_wavelet_filter_bank(
    basis: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return fixed one-level analysis and synthesis filters.

    Haar keeps the original pairwise arithmetic exactly so the frozen Haar
    checkpoint remains bitwise compatible.  The orthogonal alternatives use
    their compact reconstruction low-pass filters and quadrature-mirror
    high-pass filters in the positive-index correlation/scatter convention used
    below.  Bior4.4 instead uses its distinct reversed decomposition filters for
    analysis and reconstruction filters for synthesis; applying the orthogonal
    reversal rule to this basis would not be a valid dual filter bank.

    All tensors are non-trainable buffers.  The coefficient values follow the
    canonical PyWavelets 1.6 filter banks, but PyWavelets is not a runtime
    dependency.
    """
    normalized = basis.lower()
    if normalized == "haar":
        analysis_low = torch.tensor([0.5, 0.5], dtype=torch.float32)
        analysis_high = torch.tensor([0.5, -0.5], dtype=torch.float32)
        synthesis_low = torch.tensor([1.0, 1.0], dtype=torch.float32)
        synthesis_high = torch.tensor([1.0, -1.0], dtype=torch.float32)
    elif normalized in {"db2", "coif1", "sym4", "db6"}:
        lowpass_coefficients = {
            "db2": (
                0.48296291314453416,
                0.8365163037378079,
                0.2241438680420134,
                -0.12940952255126037,
            ),
            "coif1": (
                -0.07273261951252645,
                0.3378976624574818,
                0.8525720202116004,
                0.3848648468648578,
                -0.07273261951252645,
                -0.015655728135791993,
            ),
            "sym4": (
                0.0322231006040427,
                -0.012603967262037833,
                -0.09921954357684722,
                0.29785779560527736,
                0.8037387518059161,
                0.49761866763201545,
                -0.02963552764599851,
                -0.07576571478927333,
            ),
            "db6": (
                0.11154074335010947,
                0.49462389039845306,
                0.7511339080210954,
                0.31525035170919763,
                -0.22626469396543983,
                -0.12976686756726194,
                0.09750160558732304,
                0.027522865530305727,
                -0.03158203931748603,
                0.0005538422011614961,
                0.004777257510945511,
                -0.0010773010853084796,
            ),
        }
        analysis_low = torch.tensor(
            lowpass_coefficients[normalized],
            dtype=torch.float32,
        )
        alternating_sign = torch.ones_like(analysis_low)
        alternating_sign[1::2] = -1.0
        analysis_high = analysis_low.flip(0) * alternating_sign
        synthesis_low = analysis_low.clone()
        synthesis_high = analysis_high.clone()
    elif normalized == "bior4.4":
        analysis_low = torch.tensor(
            (
                0.03782845550726404,
                -0.023849465019556843,
                -0.11062440441843718,
                0.37740285561283066,
                0.8526986790088938,
                0.37740285561283066,
                -0.11062440441843718,
                -0.023849465019556843,
                0.03782845550726404,
                0.0,
            ),
            dtype=torch.float32,
        )
        analysis_high = torch.tensor(
            (
                0.0,
                0.0,
                -0.06453888262869706,
                0.04068941760916406,
                0.41809227322161724,
                -0.7884856164055829,
                0.41809227322161724,
                0.04068941760916406,
                -0.06453888262869706,
                0.0,
            ),
            dtype=torch.float32,
        )
        synthesis_low = torch.tensor(
            (
                0.0,
                -0.06453888262869706,
                -0.04068941760916406,
                0.41809227322161724,
                0.7884856164055829,
                0.41809227322161724,
                -0.04068941760916406,
                -0.06453888262869706,
                0.0,
                0.0,
            ),
            dtype=torch.float32,
        )
        synthesis_high = torch.tensor(
            (
                0.0,
                -0.03782845550726404,
                -0.023849465019556843,
                0.11062440441843718,
                0.37740285561283066,
                -0.8526986790088938,
                0.37740285561283066,
                0.11062440441843718,
                -0.023849465019556843,
                -0.03782845550726404,
            ),
            dtype=torch.float32,
        )
    else:
        choices = ", ".join(SUPPORTED_WAVELET_BASES)
        raise ValueError(f"wavelet_basis must be one of: {choices}")
    return analysis_low, analysis_high, synthesis_low, synthesis_high


class _WEARecLayer(nn.Module):
    """Original real-valued dynamic frequency and Haar wavelet layer."""

    def __init__(
        self,
        args,
        combine_mode="gate",
        num_heads=4,
        dropout=0.1,
        adaptive=True,
    ):
        super().__init__()
        self.complex_weight = nn.Parameter(
            torch.randn(
                1,
                args.max_seq_length // 2 + 1,
                args.hidden_size,
                2,
                dtype=torch.float32,
            )
            * 0.02
        )

        if args.hidden_size % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.num_heads = args.num_heads
        self.head_dim = args.hidden_size // self.num_heads
        self.seq_len = args.max_seq_length
        self.wavelet_basis = str(
            getattr(args, "wavelet_basis", "haar")
        ).lower()
        filters = fixed_wavelet_filter_bank(self.wavelet_basis)
        for name, values in zip(
            (
                "wavelet_analysis_low",
                "wavelet_analysis_high",
                "wavelet_synthesis_low",
                "wavelet_synthesis_high",
            ),
            filters,
        ):
            self.register_buffer(name, values, persistent=False)
        self.adaptive = adaptive
        self.combine_mode = "gate"
        self.freq_bins = args.max_seq_length // 2 + 1
        self.alpha = args.alpha
        self.complex_weight = nn.Parameter(
            torch.randn(
                1,
                self.num_heads,
                args.max_seq_length // 2,
                self.head_dim,
                dtype=torch.float32,
            )
            * 0.02
        )
        self.base_filter = nn.Parameter(
            torch.ones(self.num_heads, self.freq_bins, 1)
        )
        self.base_bias = nn.Parameter(
            torch.full((self.num_heads, self.freq_bins, 1), -0.1)
        )

        if adaptive:
            self.adaptive_mlp = nn.Sequential(
                nn.Linear(args.hidden_size, args.hidden_size),
                nn.GELU(),
                nn.Linear(
                    args.hidden_size,
                    self.num_heads * self.freq_bins * 2,
                ),
            )

        if self.combine_mode == "gate":
            self.gate_param = nn.Parameter(torch.tensor(0.0))
            self.proj_concat = None
        elif self.combine_mode == "concat":
            self.proj_concat = nn.Linear(
                2 * args.hidden_size,
                args.hidden_size,
            )
            self.gate_param = None
        else:
            raise ValueError(
                "combine_mode must be either 'gate' or 'concat'"
            )

        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)

    def wavelet_transform(self, x_heads):
        if self.wavelet_basis != "haar":
            return self._periodic_wavelet_transform(x_heads)
        batch, heads, length, head_dim = x_heads.shape
        even_length = length if (length % 2) == 0 else (length - 1)
        x_heads = x_heads[:, :, :even_length, :]
        x_even = x_heads[:, :, 0::2, :]
        x_odd = x_heads[:, :, 1::2, :]
        approx = 0.5 * (x_even + x_odd)
        detail = 0.5 * (x_even - x_odd)
        detail = detail * self.complex_weight
        x_even_recon = approx + detail
        x_odd_recon = approx - detail
        output = torch.zeros_like(x_heads)
        output[:, :, 0::2, :] = x_even_recon
        output[:, :, 1::2, :] = x_odd_recon
        if even_length < length:
            padding = torch.zeros(
                (batch, heads, 1, head_dim),
                device=output.device,
                dtype=output.dtype,
            )
            output = torch.cat([output, padding], dim=2)
        return output

    def _periodic_wavelet_transform(self, x_heads):
        """Apply a critically sampled fixed filter bank with periodization."""
        batch, heads, length, head_dim = x_heads.shape
        even_length = length if (length % 2) == 0 else (length - 1)
        if even_length <= 0:
            raise ValueError("wavelet input must contain at least two positions")
        x_even = x_heads[:, :, :even_length, :]
        coefficient_count = even_length // 2
        support = int(self.wavelet_analysis_low.numel())
        starts = 2 * torch.arange(
            coefficient_count,
            device=x_heads.device,
        )
        taps = torch.arange(support, device=x_heads.device)
        indices = (starts[:, None] + taps[None, :]) % even_length
        samples = x_even[:, :, indices, :]

        analysis_low = self.wavelet_analysis_low.to(
            device=x_heads.device,
            dtype=x_heads.dtype,
        ).view(1, 1, 1, support, 1)
        analysis_high = self.wavelet_analysis_high.to(
            device=x_heads.device,
            dtype=x_heads.dtype,
        ).view(1, 1, 1, support, 1)
        approx = (samples * analysis_low).sum(dim=3)
        detail = (samples * analysis_high).sum(dim=3)
        detail = detail * self.complex_weight[
            :, :, :coefficient_count, :
        ]

        synthesis_low = self.wavelet_synthesis_low.to(
            device=x_heads.device,
            dtype=x_heads.dtype,
        ).view(1, 1, 1, support, 1)
        synthesis_high = self.wavelet_synthesis_high.to(
            device=x_heads.device,
            dtype=x_heads.dtype,
        ).view(1, 1, 1, support, 1)
        contributions = (
            approx.unsqueeze(3) * synthesis_low
            + detail.unsqueeze(3) * synthesis_high
        ).reshape(batch, heads, coefficient_count * support, head_dim)
        scatter_indices = indices.reshape(1, 1, -1, 1).expand(
            batch,
            heads,
            -1,
            head_dim,
        )
        output = torch.zeros_like(x_even).scatter_add(
            2,
            scatter_indices,
            contributions,
        )
        if even_length < length:
            output = torch.cat(
                [
                    output,
                    torch.zeros(
                        (batch, heads, 1, head_dim),
                        device=output.device,
                        dtype=output.dtype,
                    ),
                ],
                dim=2,
            )
        return output

    def forward(self, input_tensor):
        batch, seq_len, hidden = input_tensor.shape
        x_heads = input_tensor.view(
            batch,
            seq_len,
            self.num_heads,
            self.head_dim,
        ).permute(0, 2, 1, 3)
        frequency = torch.fft.rfft(x_heads, dim=2, norm="ortho")

        if self.adaptive:
            context = input_tensor.mean(dim=1)
            adapt_params = self.adaptive_mlp(context).view(
                batch,
                self.num_heads,
                self.freq_bins,
                2,
            )
            adaptive_scale = adapt_params[..., 0:1]
            adaptive_bias = adapt_params[..., 1:2]
        else:
            adaptive_scale = torch.zeros(
                batch,
                self.num_heads,
                self.freq_bins,
                1,
                device=input_tensor.device,
            )
            adaptive_bias = torch.zeros(
                batch,
                self.num_heads,
                self.freq_bins,
                1,
                device=input_tensor.device,
            )

        effective_filter = self.base_filter * (1 + adaptive_scale)
        effective_bias = self.base_bias + adaptive_bias
        frequency_mod = frequency * effective_filter + effective_bias
        x_fft = torch.fft.irfft(
            frequency_mod,
            dim=2,
            n=self.seq_len,
            norm="ortho",
        )
        x_wavelet = self.wavelet_transform(x_heads)

        if self.combine_mode == "gate":
            x_combined = (
                (1.0 - self.alpha) * x_wavelet
                + self.alpha * x_fft
            )
        else:
            x_wavelet_reshaped = x_wavelet.permute(
                0,
                2,
                1,
                3,
            ).reshape(batch, seq_len, hidden)
            x_fft_reshaped = x_fft.permute(0, 2, 1, 3).reshape(
                batch,
                seq_len,
                hidden,
            )
            x_combined = self.proj_concat(
                torch.cat(
                    [x_wavelet_reshaped, x_fft_reshaped],
                    dim=-1,
                )
            )
            x_combined = x_combined.view(
                batch,
                seq_len,
                self.num_heads,
                self.head_dim,
            ).permute(0, 2, 1, 3)

        x_out = x_combined.permute(0, 2, 1, 3).reshape(
            batch,
            seq_len,
            hidden,
        )
        hidden_states = self.out_dropout(x_out)
        return self.LayerNorm(hidden_states + input_tensor)


class _WEARecBlock(nn.Module):
    """One original WEARec layer followed by its feed-forward network."""

    def __init__(self, args):
        super().__init__()
        self.layer = _WEARecLayer(args)
        self.feed_forward = FeedForward(args)

    def forward(self, hidden_states):
        return self.feed_forward(self.layer(hidden_states))


class _WEARecEncoder(nn.Module):
    """Original stacked WEARec encoder."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        block = _WEARecBlock(args)
        self.blocks = nn.ModuleList(
            [copy.deepcopy(block) for _ in range(args.num_hidden_layers)]
        )

    def forward(self, hidden_states, output_all_encoded_layers=False):
        all_encoder_layers = [hidden_states]
        for layer_module in self.blocks:
            hidden_states = layer_module(hidden_states)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class _TCFBackbone(SequentialRecModel):
    """TCF with temporal-state embeddings."""

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.time_embeddings = nn.Embedding(
            args.num_time_span_ids,
            args.hidden_size,
            padding_idx=0,
        )
        self.time_embedding_scale = getattr(
            args,
            "time_embedding_scale",
            1.0,
        )
        self.temporal_padding_is_neutral = True
        self.item_encoder = _WEARecEncoder(args)
        self.apply(self.init_weights)
        with torch.no_grad():
            self.time_embeddings.weight[0].zero_()

    def add_time_embedding(self, sequence, ts_seq):
        item_embeddings = self.item_embeddings(sequence)
        time_embeddings = self.time_embeddings(ts_seq)
        time_embeddings = time_embeddings.masked_fill(
            ts_seq.eq(0).unsqueeze(-1),
            0.0,
        )
        if self.time_embedding_scale != 1.0:
            time_embeddings = (
                time_embeddings * self.time_embedding_scale
            )
        sequence_emb = item_embeddings + time_embeddings

        seq_length = sequence.size(1)
        position_ids = torch.arange(
            seq_length,
            dtype=torch.long,
            device=sequence.device,
        )
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = sequence_emb + position_embeddings

        sequence_emb = self.LayerNorm(sequence_emb)
        return self.dropout(sequence_emb)

    def forward(
        self,
        input_ids,
        ts_seq=None,
        user_ids=None,
        all_sequence_output=False,
    ):
        if ts_seq is not None:
            sequence_emb = self.add_time_embedding(input_ids, ts_seq)
        else:
            sequence_emb = self.add_position_embedding(input_ids)
        item_encoded_layers = self.item_encoder(
            sequence_emb,
            output_all_encoded_layers=True,
        )
        if all_sequence_output:
            return item_encoded_layers
        return item_encoded_layers[-1]

    def calculate_loss(
        self,
        input_ids,
        answers,
        neg_answers,
        same_target,
        user_ids,
        ts_seq=None,
    ):
        seq_output = self.forward(input_ids, ts_seq)
        seq_output = seq_output[:, -1, :]
        item_emb = self.item_embeddings.weight
        logits = torch.matmul(seq_output, item_emb.transpose(0, 1))
        return nn.CrossEntropyLoss()(logits, answers)

    def predict(self, input_ids, user_ids, ts_seq=None):
        return self.forward(input_ids, ts_seq, user_ids)


def build_markov_transition_index(
    user_seq: list[list[int]],
    max_len: int,
    mode: str = "last",
    alpha: float = 2.0,
) -> dict[int, tuple[list[int], list[float]]]:
    """Build a transition table using truncated training prefixes only."""
    if mode not in {"last", "window"}:
        raise ValueError("mode must be 'last' or 'window'")
    if max_len <= 0:
        raise ValueError("max_len must be positive")
    if alpha <= 0:
        raise ValueError("alpha must be positive")

    counts: dict[int, dict[int, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for sequence in user_seq:
        train_items = sequence[-(max_len + 2):-2]
        if len(train_items) < 2:
            continue
        if mode == "last":
            for source, target in zip(
                train_items[:-1],
                train_items[1:],
            ):
                counts[source][target] += 1.0
        else:
            for target_index in range(1, len(train_items)):
                target = train_items[target_index]
                start = max(0, target_index - max_len)
                for source_index in range(start, target_index):
                    distance = target_index - source_index
                    counts[train_items[source_index]][target] += math.exp(
                        -float(distance - 1) / alpha
                    )

    scores: dict[int, tuple[list[int], list[float]]] = {}
    for source, target_counts in counts.items():
        if not target_counts:
            continue
        max_count = max(target_counts.values())
        target_items = []
        values = []
        for target, value in target_counts.items():
            target_items.append(target)
            values.append(math.log1p(value) / math.log1p(max_count))
        scores[source] = (target_items, values)
    return scores


class _TCFCalibratedBackbone(_TCFBackbone):
    """TCF plus a non-trainable, train-prefix transition prior."""

    def __init__(self, args):
        super().__init__(args)
        self.markov_lambda = float(
            getattr(args, "markov_lambda", 0.0)
        )
        self.markov_transition_window = int(
            getattr(args, "markov_transition_window", 3)
        )
        self.markov_transition_alpha = float(
            getattr(args, "markov_transition_alpha", 2.0)
        )
        self.markov_recency_tau = float(
            getattr(args, "markov_recency_tau", 1.5)
        )
        self.markov_transition_mode = getattr(
            args,
            "markov_transition_mode",
            "last",
        )

        if self.markov_transition_window <= 0:
            raise ValueError(
                "markov_transition_window must be positive"
            )
        if self.markov_transition_alpha <= 0:
            raise ValueError("markov_transition_alpha must be positive")
        if self.markov_recency_tau <= 0:
            raise ValueError("markov_recency_tau must be positive")
        if self.markov_transition_mode not in {"last", "window"}:
            raise ValueError(
                "markov_transition_mode must be 'last' or 'window'"
            )

        self._markov_transition_scores: dict[
            int,
            tuple[list[int], list[float]],
        ] = {}
        self._markov_sparse_cache: dict[
            tuple[str, torch.dtype, int],
            torch.Tensor,
        ] = {}

    def needs_markov_transition_index(self):
        return self.markov_lambda != 0.0

    def set_markov_transition_index(self, user_seq: list[list[int]]):
        self._markov_transition_scores = build_markov_transition_index(
            user_seq,
            self.args.max_seq_length,
            self.markov_transition_mode,
            self.markov_transition_alpha,
        )
        self._markov_sparse_cache.clear()
        transition_count = sum(
            len(target_items)
            for target_items, _ in self._markov_transition_scores.values()
        )
        return {
            "source_items": len(self._markov_transition_scores),
            "transitions": transition_count,
        }

    def _transition_sparse_matrix(
        self,
        item_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        cache_key = (str(device), dtype, item_size)
        if cache_key in self._markov_sparse_cache:
            return self._markov_sparse_cache[cache_key]

        source_indices = []
        target_indices = []
        values = []
        for source, (
            target_items,
            row_values,
        ) in self._markov_transition_scores.items():
            if source >= item_size:
                continue
            for target, value in zip(target_items, row_values):
                if target >= item_size:
                    continue
                source_indices.append(source)
                target_indices.append(target)
                values.append(value)

        if values:
            indices = torch.tensor(
                [source_indices, target_indices],
                device=device,
                dtype=torch.long,
            )
            value_tensor = torch.tensor(
                values,
                device=device,
                dtype=dtype,
            )
        else:
            indices = torch.empty(
                (2, 0),
                device=device,
                dtype=torch.long,
            )
            value_tensor = torch.empty(
                (0,),
                device=device,
                dtype=dtype,
            )

        matrix = torch.sparse_coo_tensor(
            indices,
            value_tensor,
            size=(item_size, item_size),
            device=device,
            dtype=dtype,
        ).coalesce()
        self._markov_sparse_cache[cache_key] = matrix
        return matrix

    def markov_transition_logits(
        self,
        input_ids: torch.Tensor,
        item_size: int,
    ):
        if self.markov_lambda == 0.0:
            return input_ids.new_zeros(
                (input_ids.size(0), item_size),
                dtype=self.item_embeddings.weight.dtype,
            )
        if not self._markov_transition_scores:
            raise RuntimeError(
                "Markov transition index is empty. Call "
                "set_markov_transition_index(user_seq) before evaluation."
            )

        device = input_ids.device
        dtype = self.item_embeddings.weight.dtype
        source_weights = torch.zeros(
            input_ids.size(0),
            item_size,
            device=device,
            dtype=dtype,
        )

        valid_mask = input_ids.ne(0)
        valid_count = valid_mask.sum(dim=1)
        cumulative = valid_mask.to(dtype=torch.long).cumsum(dim=1)
        distance = valid_count.unsqueeze(1) - cumulative
        recent_mask = (
            valid_mask
            & distance.ge(0)
            & distance.lt(self.markov_transition_window)
        )
        recency_weights = torch.exp(
            -distance.to(dtype=dtype) / self.markov_recency_tau
        )
        recency_weights = (
            recency_weights * recent_mask.to(dtype=dtype)
        )

        batch_indices = torch.arange(
            input_ids.size(0),
            device=device,
        ).unsqueeze(1)
        source_weights.index_put_(
            (batch_indices.expand_as(input_ids), input_ids),
            recency_weights,
            accumulate=True,
        )

        transition_matrix = self._transition_sparse_matrix(
            item_size,
            device,
            dtype,
        )
        scores = torch.sparse.mm(
            transition_matrix.transpose(0, 1),
            source_weights.transpose(0, 1),
        ).transpose(0, 1)
        return scores * self.markov_lambda

    def full_sort_predict(self, input_ids, user_ids=None, ts_seq=None):
        sequence_output = self.forward(
            input_ids,
            ts_seq=ts_seq,
            user_ids=user_ids,
        )
        last_output = sequence_output[:, -1, :]
        base_logits = torch.matmul(
            last_output,
            self.item_embeddings.weight.transpose(0, 1),
        )
        return base_logits + self.markov_transition_logits(
            input_ids,
            base_logits.size(1),
        )


class ComplexDFFLayer(_WEARecLayer):
    """Add amplitude/phase modulation and optional low-rank channel mixing."""

    def __init__(self, args):
        super().__init__(args)
        self.complex_dff_epsilon = float(
            getattr(args, "complex_dff_epsilon", 0.05)
        )
        self.complex_dff_rank = int(
            getattr(args, "complex_dff_rank", 0)
        )
        self.complex_dff_log_amp_limit = float(
            getattr(args, "complex_dff_log_amp_limit", 2.0)
        )
        self.complex_dff_phase_limit = float(
            getattr(args, "complex_dff_phase_limit", math.pi)
        )
        self.complex_dff_disable_amplitude = bool(
            getattr(args, "complex_dff_disable_amplitude", False)
        )
        self.complex_dff_disable_phase = bool(
            getattr(args, "complex_dff_disable_phase", False)
        )

        if self.complex_dff_epsilon < 0:
            raise ValueError("complex_dff_epsilon must be non-negative")
        if self.complex_dff_rank < 0:
            raise ValueError("complex_dff_rank must be non-negative")
        if self.complex_dff_rank > self.head_dim:
            raise ValueError(
                "complex_dff_rank cannot exceed per-head channel dimension"
            )
        if self.complex_dff_log_amp_limit <= 0:
            raise ValueError("complex_dff_log_amp_limit must be positive")
        if self.complex_dff_phase_limit <= 0:
            raise ValueError("complex_dff_phase_limit must be positive")

        self.complex_dff_mlp = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.GELU(),
            nn.Linear(
                args.hidden_size,
                self.num_heads * self.freq_bins * 2,
            ),
        )

        if self.complex_dff_rank:
            scale = float(getattr(args, "initializer_range", 0.02))
            self.complex_dff_channel_down = nn.Parameter(
                torch.randn(
                    self.num_heads,
                    self.freq_bins,
                    self.complex_dff_rank,
                    self.head_dim,
                    2,
                    dtype=torch.float32,
                )
                * scale
            )
            self.complex_dff_channel_up = nn.Parameter(
                torch.zeros(
                    self.num_heads,
                    self.freq_bins,
                    self.head_dim,
                    self.complex_dff_rank,
                    2,
                    dtype=torch.float32,
                )
            )
        else:
            self.register_parameter("complex_dff_channel_down", None)
            self.register_parameter("complex_dff_channel_up", None)

    def reset_near_identity_parameters(self) -> None:
        """Make the adaptive gain amplitude=1, phase=0 at initialization."""
        output = self.complex_dff_mlp[-1]
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        if self.complex_dff_channel_up is not None:
            nn.init.zeros_(self.complex_dff_channel_up)

    def complex_gain(
        self,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return complex gain, bounded log-amplitude, and bounded phase."""
        batch = context.size(0)
        raw = self.complex_dff_mlp(context).view(
            batch,
            self.num_heads,
            self.freq_bins,
            2,
        )
        if self.complex_dff_disable_amplitude:
            delta_log_amplitude = torch.zeros_like(raw[..., 0:1])
        else:
            delta_log_amplitude = (
                self.complex_dff_log_amp_limit
                * torch.tanh(raw[..., 0:1])
            )
        if self.complex_dff_disable_phase:
            delta_phase = torch.zeros_like(raw[..., 1:2])
        else:
            delta_phase = (
                self.complex_dff_phase_limit
                * torch.tanh(raw[..., 1:2])
            )
        amplitude = torch.exp(delta_log_amplitude)
        gain = torch.polar(amplitude, delta_phase)
        return gain, delta_log_amplitude, delta_phase

    def low_rank_channel_residual(
        self,
        frequency: torch.Tensor,
    ) -> torch.Tensor:
        """Mix ``head_dim`` channels with a per-head/frequency complex rank."""
        if self.complex_dff_rank == 0:
            return torch.zeros_like(frequency)
        down = torch.view_as_complex(
            self.complex_dff_channel_down.contiguous()
        )
        up = torch.view_as_complex(
            self.complex_dff_channel_up.contiguous()
        )
        latent = torch.einsum("bhfd,hfrd->bhfr", frequency, down)
        mixed = torch.einsum("bhfr,hfdr->bhfd", latent, up)
        return mixed / math.sqrt(self.complex_dff_rank)

    def _legacy_frequency_filter(
        self,
        input_tensor: torch.Tensor,
        frequency: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = input_tensor.size(0)
        if self.adaptive:
            context = input_tensor.mean(dim=1)
            adapt_params = self.adaptive_mlp(context).view(
                batch,
                self.num_heads,
                self.freq_bins,
                2,
            )
            adaptive_scale = adapt_params[..., 0:1]
            adaptive_bias = adapt_params[..., 1:2]
        else:
            context = input_tensor.mean(dim=1)
            adaptive_scale = torch.zeros(
                batch,
                self.num_heads,
                self.freq_bins,
                1,
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            )
            adaptive_bias = torch.zeros_like(adaptive_scale)

        effective_filter = self.base_filter * (1 + adaptive_scale)
        effective_bias = self.base_bias + adaptive_bias
        return frequency * effective_filter + effective_bias, context

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden = input_tensor.shape
        if seq_len != self.seq_len:
            raise ValueError(
                f"expected sequence length {self.seq_len}, got {seq_len}"
            )
        x_heads = input_tensor.view(
            batch,
            seq_len,
            self.num_heads,
            self.head_dim,
        ).permute(0, 2, 1, 3)
        frequency = torch.fft.rfft(x_heads, dim=2, norm="ortho")
        legacy_frequency, context = self._legacy_frequency_filter(
            input_tensor,
            frequency,
        )

        gain, _, _ = self.complex_gain(context)
        multiplier = 1.0 + self.complex_dff_epsilon * gain
        frequency_mod = legacy_frequency * multiplier
        if self.complex_dff_rank:
            frequency_mod = (
                frequency_mod
                + self.complex_dff_epsilon
                * self.low_rank_channel_residual(legacy_frequency)
            )

        x_fft = torch.fft.irfft(
            frequency_mod,
            dim=2,
            n=self.seq_len,
            norm="ortho",
        )
        x_wavelet = self.wavelet_transform(x_heads)
        if self.combine_mode == "gate":
            x_combined = (
                (1.0 - self.alpha) * x_wavelet
                + self.alpha * x_fft
            )
        else:
            x_wavelet = x_wavelet.permute(0, 2, 1, 3).reshape(
                batch,
                seq_len,
                hidden,
            )
            x_fft = x_fft.permute(0, 2, 1, 3).reshape(
                batch,
                seq_len,
                hidden,
            )
            x_combined = self.proj_concat(
                torch.cat([x_wavelet, x_fft], dim=-1)
            )
            x_combined = x_combined.view(
                batch,
                seq_len,
                self.num_heads,
                self.head_dim,
            ).permute(0, 2, 1, 3)

        x_out = x_combined.permute(0, 2, 1, 3).reshape(
            batch,
            seq_len,
            hidden,
        )
        hidden_states = self.out_dropout(x_out)
        return self.LayerNorm(hidden_states + input_tensor)


class ComplexDFFBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.layer = ComplexDFFLayer(args)
        self.feed_forward = FeedForward(args)

    def forward(self, hidden_states):
        return self.feed_forward(self.layer(hidden_states))


class ComplexDFFEncoder(_WEARecEncoder):
    def __init__(self, args):
        nn.Module.__init__(self)
        self.args = args
        block = ComplexDFFBlock(args)
        self.blocks = nn.ModuleList(
            [copy.deepcopy(block) for _ in range(args.num_hidden_layers)]
        )

    def reset_near_identity_parameters(self) -> None:
        for block in self.blocks:
            block.layer.reset_near_identity_parameters()


class _ComplexDFFModelMixin:
    complex_frequency_filter = True

    def __init__(self, args):
        super().__init__(args)
        self.item_encoder = ComplexDFFEncoder(args)
        self.item_encoder.apply(self.init_weights)
        self.item_encoder.reset_near_identity_parameters()

    def load_state_dict(self, state_dict, strict=True):
        result = super().load_state_dict(state_dict, strict=strict)
        if strict:
            return result
        # ``main.py`` audits missing keys during paired WEARec initialization.
        # The isolated complex-DFF entrypoint deliberately adds only these
        # parameters, so hide them from that legacy audit while retaining all
        # other missing/unexpected keys for the original safety checks.
        missing = [
            key
            for key in result.missing_keys
            if "complex_dff" not in key
        ]
        return type(result)(missing, result.unexpected_keys)


class TCFModel(
    _ComplexDFFModelMixin,
    _TCFCalibratedBackbone,
):
    """The complete final TCF model used for training and evaluation."""

__all__ = [
    "SUPPORTED_WAVELET_BASES",
    "TCFModel",
    "build_markov_transition_index",
]
