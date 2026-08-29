import os
import torch
import numpy as np

from model import MODEL_DICT
from trainers import Trainer
from utils import EarlyStopping, check_path, set_seed, parse_args, set_logger
from dataset import get_seq_dic, get_dataloder, get_rating_matrix

def main():

    args = parse_args()
    log_path = os.path.join(args.output_dir, args.train_name + '.log')
    logger = set_logger(log_path)

    set_seed(args.seed)
    check_path(args.output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    seq_dic, max_item, num_users = get_seq_dic(args)
    args.item_size = max_item + 1
    args.num_users = num_users + 1

    args.checkpoint_path = os.path.join(args.output_dir, args.train_name + '.pt')
    args.same_target_path = os.path.join(args.data_dir, args.data_name+'_same_target.npy')
    train_dataloader, eval_dataloader, test_dataloader = get_dataloder(args,seq_dic)

    logger.info(str(args))
    model_type = args.model_type.lower()
    paired_base_models = {
        "tcf": "wearec",
        "tcf_time": "tcf",
        "tcf_static_gate": "tcf",
        "tcf_selective_gate": "tcf",
        "tcf_residual_gate": "tcf",
        "tcf_time_static_gate": "tcf_time",
        "tcf_time_selective_gate": "tcf_time",
        "tcf_time_residual_gate": "tcf_time",
        "tcf_time_short_history": "tcf_time",
        "tcf_time_short_intent": "tcf_time",
        "tcf_time_short_memory": "tcf_time",
        "tcf_time_post_short_memory": "tcf_time",
        "tcf_time_short_score": "tcf_time",
        "tcf_time_transition": "tcf_time",
        "tcf_time_markov": "tcf_time",
        "tcf_time_tg": "tcf_time",
    }
    if args.paired_time_factorial and model_type not in {"tcf", "tcf_time"}:
        raise ValueError(
            "--paired_time_factorial is valid only for TCF training"
        )
    paired_initialization = (
        args.paired_gate_experiment or args.paired_time_factorial
    )

    if paired_initialization and model_type in paired_base_models:
        # Build the matching baseline first, exactly as the original main path
        # does. Save the post-construction RNG state so RandomSampler and
        # dropout follow the same trajectory in both baseline and gated runs.
        initial_rng_state = torch.get_rng_state()
        base_model_type = paired_base_models[model_type]
        base_model = MODEL_DICT[base_model_type](args=args)
        paired_training_rng_state = torch.get_rng_state()

        # Build the gated model reproducibly, copy all shared TCF
        # parameters, then restore the baseline's training RNG state. The gate
        # keeps its fixed-alpha initialization without perturbing sampling or
        # dropout randomness.
        torch.set_rng_state(initial_rng_state)
        model = MODEL_DICT[model_type](args=args)
        load_result = model.load_state_dict(base_model.state_dict(), strict=False)
        torch.set_rng_state(paired_training_rng_state)
        if load_result.unexpected_keys:
            raise RuntimeError(
                "Unexpected keys while pairing gate initialization: "
                f"{load_result.unexpected_keys}"
            )
        allowed_missing_markers = (
            "gate",
            "short",
            "transition",
            "layer_scale",
            "time_embeddings",
        )
        if any(
            not any(marker in key for marker in allowed_missing_markers)
            for key in load_result.missing_keys
        ):
            raise RuntimeError(
                "Non-gate/non-short parameters were missing during paired "
                "initialization: "
                f"{load_result.missing_keys}"
            )
        logger.info(
            f"Paired initialization: {model_type} shares all common "
            f"parameters with {base_model_type}"
        )
        del base_model
    else:
        # This is intentionally the unchanged original construction path.
        # In particular, a paired baseline must not reseed after construction.
        model = MODEL_DICT[model_type](args=args)

    if getattr(model, "temporal_padding_is_neutral", False):
        logger.info(
            "Temporal padding invariant: ts_seq=0 contributes an exact "
            "zero time embedding"
        )

    if (
        hasattr(model, "needs_markov_transition_index")
        and model.needs_markov_transition_index()
    ):
        markov_stats = model.set_markov_transition_index(seq_dic["user_seq"])
        logger.info(f"Markov transition index: {markov_stats}")

    logger.info(model)
    trainer = Trainer(model, train_dataloader, eval_dataloader, test_dataloader, args, logger)

    args.valid_rating_matrix, args.test_rating_matrix = get_rating_matrix(args.data_name, seq_dic, max_item)

    if paired_initialization:
        logger.info(
            "Paired sampling/dropout RNG aligned with the original baseline "
            "post-construction state"
        )

    if args.do_eval:
        if args.load_model is None:
            logger.info(f"No model input!")
            exit(0)
        else:
            args.checkpoint_path = os.path.join(args.output_dir, args.load_model + '.pt')
            trainer.load(args.checkpoint_path)
            logger.info(
                f"Load model from {args.checkpoint_path} "
                f"for {args.eval_split}!"
            )
            if args.eval_split == "valid":
                scores, result_info = trainer.valid(0)
            else:
                scores, result_info = trainer.test(0)

    else:
        early_stopping = EarlyStopping(args.checkpoint_path, logger=logger, patience=args.patience, verbose=True)
        for epoch in range(args.epochs):

            trainer.train(epoch)
            scores, _ = trainer.valid(epoch)
            # evaluate on MRR
            early_stopping(np.array(scores[-1:]), trainer.model)
            if early_stopping.early_stop:
                logger.info("Early stopping")
                break

        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        if args.skip_test_after_training:
            result_info = {
                "best_development_NDCG@20": float(early_stopping.best_score[-1]),
                "test_evaluated": False,
            }
            logger.info(
                "Test evaluation skipped until checkpoint selection completes."
            )
        else:
            logger.info("---------------Test Score---------------")
            scores, result_info = trainer.test(0)

    logger.info(args.train_name)
    logger.info(result_info)


main()
