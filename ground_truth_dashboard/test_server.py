import unittest
from pathlib import Path

from server import CelebAGroundTruth, DashboardPaths, parse_query


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class GroundTruthDashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repository = CelebAGroundTruth(
            DashboardPaths(
                PROJECT_ROOT / "celeba_evaluation.json",
                PROJECT_ROOT / "celeba",
            )
        )

    def test_query_parser(self):
        self.assertEqual(
            parse_query("+Black_Hair, -Wavy_Hair"),
            [
                {"operator": "+", "attribute": "Black_Hair", "expected": 1},
                {"operator": "-", "attribute": "Wavy_Hair", "expected": -1},
            ],
        )

    def test_pytorch_test_index_maps_to_real_filename(self):
        self.assertEqual(self.repository.test_filenames[13], "182651.jpg")
        self.assertNotEqual(self.repository.test_filenames[13], "000013.jpg")

    def test_first_smiling_entry(self):
        entry = self.repository.entry(query_index=0, source_index=13)
        self.assertEqual(entry["source"]["filename"], "182651.jpg")
        self.assertEqual(entry["targets"][0]["dataset_index"], 325)
        self.assertEqual(entry["targets"][0]["filename"], "182963.jpg")
        self.assertEqual(len(entry["targets"]), entry["available_target_count"])
        self.assertTrue(
            all(target["query_checks"][0]["satisfied"] for target in entry["targets"])
        )
        self.assertTrue(all(target["hamming_distance"] <= 2 for target in entry["targets"]))

    def test_same_identity_targets_are_reported(self):
        entry = self.repository.entry(query_index=0, source_index=28)
        self.assertEqual(entry["same_identity_target_count"], 4)
        self.assertEqual(entry["same_identity_target_ranks"], [1, 2, 3, 9])


if __name__ == "__main__":
    unittest.main()
