#!/usr/bin/env python3
"""Run the frozen main pipeline with isolated complex-DFF model bindings."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

import torch


SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import model  # noqa: E402
import trainers  # noqa: E402
import utils  # noqa: E402
from model.tcf import (  # noqa: E402
    SUPPORTED_WAVELET_BASES,
    TCFModel,
)


def extract_complex_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--complex_dff_epsilon", default=0.05, type=float)
    parser.add_argument("--complex_dff_rank", default=0, type=int)
    parser.add_argument(
        "--wavelet_basis",
        choices=SUPPORTED_WAVELET_BASES,
        default="haar",
        help="fixed one-level wavelet basis used alongside the CDF branch",
    )
    parser.add_argument("--complex_dff_log_amp_limit", default=2.0, type=float)
    parser.add_argument(
        "--complex_dff_phase_limit",
        default=3.141592653589793,
        type=float,
    )
    parser.add_argument(
        "--complex_dff_disable_amplitude",
        action="store_true",
        help="fix the adaptive amplitude multiplier to one exactly",
    )
    parser.add_argument(
        "--complex_dff_disable_phase",
        action="store_true",
        help="fix the adaptive phase offset to zero exactly",
    )
    parser.add_argument("--complex_dff_warmstart", type=Path)
    parser.add_argument(
        "--complex_dff_train_scope",
        choices=("all", "complex-only"),
        default="all",
    )
    parser.add_argument(
        "--complex_dff_validate_initial",
        action="store_true",
    )
    complex_args, remaining = parser.parse_known_args()
    if complex_args.complex_dff_epsilon < 0:
        raise ValueError("complex_dff_epsilon must be non-negative")
    if complex_args.complex_dff_rank < 0:
        raise ValueError("complex_dff_rank must be non-negative")
    if (
        complex_args.complex_dff_train_scope == "complex-only"
        and complex_args.complex_dff_warmstart is None
    ):
        raise ValueError(
            "complex-only training requires --complex_dff_warmstart"
        )
    if complex_args.complex_dff_warmstart is not None:
        complex_args.complex_dff_warmstart = (
            complex_args.complex_dff_warmstart.resolve()
        )
        if not complex_args.complex_dff_warmstart.is_file():
            raise FileNotFoundError(complex_args.complex_dff_warmstart)
    sys.argv = [sys.argv[0], *remaining]
    return complex_args


def bind_parse_args(complex_args: argparse.Namespace) -> None:
    original_parse_args = utils.parse_args

    def parse_args_with_complex_fields():
        args = original_parse_args()
        for name, value in vars(complex_args).items():
            setattr(args, name, value)
        return args

    utils.parse_args = parse_args_with_complex_fields


def bind_warmstart_trainer(complex_args: argparse.Namespace) -> None:
    original_init = trainers.Trainer.__init__
    if complex_args.complex_dff_warmstart is None:
        def fresh_init(
            self,
            model_instance,
            train_dataloader,
            eval_dataloader,
            test_dataloader,
            args,
            logger,
        ):
            if args.load_model is not None or args.do_eval:
                raise RuntimeError(
                    "fresh Complex DFF training requires load_model=None "
                    "and do_eval=False"
                )
            frozen = [
                name
                for name, parameter in model_instance.named_parameters()
                if not parameter.requires_grad
            ]
            if frozen:
                raise RuntimeError(
                    "fresh Complex DFF training found frozen parameters: "
                    f"{frozen}"
                )
            logger.info(
                "Complex DFF fresh training audit: "
                "load_model=None, warmstart=None, scope=all, "
                f"trainable_tensors="
                f"{sum(1 for _ in model_instance.parameters())}, "
                f"trainable_parameters="
                f"{sum(parameter.numel() for parameter in model_instance.parameters())}"
            )
            original_init(
                self,
                model_instance,
                train_dataloader,
                eval_dataloader,
                test_dataloader,
                args,
                logger,
            )

        trainers.Trainer.__init__ = fresh_init
        return

    original_train = trainers.Trainer.train

    def warmstart_init(
        self,
        model_instance,
        train_dataloader,
        eval_dataloader,
        test_dataloader,
        args,
        logger,
    ):
        checkpoint = torch.load(
            complex_args.complex_dff_warmstart,
            map_location="cpu",
        )
        current = model_instance.state_dict()
        unexpected = sorted(set(checkpoint) - set(current))
        shape_mismatch = sorted(
            key
            for key in set(checkpoint) & set(current)
            if checkpoint[key].shape != current[key].shape
        )
        missing_backbone = sorted(
            key
            for key in current
            if key not in checkpoint and "complex_dff_" not in key
        )
        if unexpected or shape_mismatch or missing_backbone:
            raise RuntimeError(
                "invalid Complex DFF warm-start checkpoint: "
                f"unexpected={unexpected}, "
                f"shape_mismatch={shape_mismatch}, "
                f"missing_backbone={missing_backbone}"
            )
        model_instance.load_state_dict(checkpoint, strict=False)
        if complex_args.complex_dff_train_scope == "complex-only":
            for name, parameter in model_instance.named_parameters():
                parameter.requires_grad = "complex_dff_" in name
            trainable = [
                name
                for name, parameter in model_instance.named_parameters()
                if parameter.requires_grad
            ]
            if not trainable:
                raise RuntimeError("no Complex DFF parameters are trainable")
        else:
            trainable = [
                name
                for name, parameter in model_instance.named_parameters()
                if parameter.requires_grad
            ]
        logger.info(
            "Complex DFF warm start: "
            f"checkpoint={complex_args.complex_dff_warmstart}, "
            f"scope={complex_args.complex_dff_train_scope}, "
            f"trainable_tensors={len(trainable)}, "
            f"trainable_parameters="
            f"{sum(parameter.numel() for parameter in model_instance.parameters() if parameter.requires_grad)}"
        )
        original_init(
            self,
            model_instance,
            train_dataloader,
            eval_dataloader,
            test_dataloader,
            args,
            logger,
        )

    def warmstart_train(self, epoch):
        if complex_args.complex_dff_validate_initial and epoch == 0:
            self.logger.info(
                "Complex DFF warm start: epoch 0 is the untouched "
                "initial checkpoint"
            )
            return
        return original_train(self, epoch)

    trainers.Trainer.__init__ = warmstart_init
    trainers.Trainer.train = warmstart_train


complex_args = extract_complex_args()
bind_parse_args(complex_args)
bind_warmstart_trainer(complex_args)
model.MODEL_DICT["tcf"] = TCFModel
runpy.run_path(str(SRC_DIR / "main.py"), run_name="__main__")
