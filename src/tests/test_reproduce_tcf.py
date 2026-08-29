from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


SRC_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scripts import reproduce_tcf  # noqa: E402


class TCFReproductionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = reproduce_tcf.read_manifest(
            reproduce_tcf.DEFAULT_MANIFEST
        )

    def test_manifest_is_fresh_training_only(self) -> None:
        training = self.manifest["training"]
        self.assertIsNone(training["load_model"])
        self.assertIsNone(training["warmstart"])
        self.assertEqual(training["train_scope"], "all")
        self.assertEqual(training["patience"], 10)
        self.assertTrue(training["skip_test_after_training"])
        self.assertEqual(
            tuple(self.manifest["datasets"]),
            reproduce_tcf.DATASET_ORDER,
        )

    def test_no_argument_entry_point_defaults_to_full_reproduction(self) -> None:
        with patch.object(sys, "argv", ["reproduce_tcf.py"]):
            args = reproduce_tcf.parse_args()
        self.assertEqual(args.mode, "full")

    def test_all_reproduction_data_are_shipped_and_aligned(self) -> None:
        audit = reproduce_tcf.data_audit(self.manifest)
        self.assertEqual(tuple(audit), reproduce_tcf.DATASET_ORDER)
        for row in audit.values():
            self.assertGreater(row["item_bytes"], 0)
            self.assertGreater(row["time_bytes"], 0)
            self.assertGreater(row["users"], 0)
            self.assertGreater(row["interactions"], 0)

    def test_result_commands_cannot_initialize_from_a_checkpoint(self) -> None:
        args = argparse.Namespace(
            output_dir=SRC_DIR / "output" / "test_reproduction_commands",
            gpu_id="0",
            no_cuda=True,
        )
        jobs = reproduce_tcf.build_jobs(
            self.manifest,
            reproduce_tcf.DATASET_ORDER,
            args,
        )
        for job in jobs:
            command = job.command()
            model_index = command.index("--model_type")
            self.assertEqual(command[model_index + 1], "tcf")
            self.assertNotIn("--load_model", command)
            self.assertNotIn("--complex_dff_warmstart", command)
            scope_index = command.index("--complex_dff_train_scope")
            self.assertEqual(command[scope_index + 1], "all")
            self.assertIn("--paired_time_factorial", command)
            self.assertIn("--skip_test_after_training", command)
            self.assertIn("--strict_full_sort_mask", command)

    def test_final_result_uses_wearec_log_layout(self) -> None:
        result = {
            "HR@10": 0.1,
            "NDCG@10": 0.2,
            "HR@20": 0.3,
            "NDCG@20": 0.4,
        }
        self.assertEqual(
            reproduce_tcf.wearec_result_info(result),
            {
                "Epoch": 0,
                "HR@10": "0.10000000",
                "NDCG@10": "0.20000000",
                "HR@20": "0.30000000",
                "NDCG@20": "0.40000000",
            },
        )

    def test_expected_metrics_and_tolerance_are_complete(self) -> None:
        tolerance = self.manifest["evaluation"][
            "verification_absolute_tolerance"
        ]
        self.assertGreater(tolerance, 0)
        self.assertLessEqual(tolerance, 0.001)
        for preset in self.manifest["datasets"].values():
            self.assertEqual(
                set(preset["expected"]), set(reproduce_tcf.METRICS)
            )


if __name__ == "__main__":
    unittest.main()
