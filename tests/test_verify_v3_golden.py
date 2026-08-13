import unittest

from scripts.verify_v3_golden import verify


def _metrics(f1=0.5, recall=0.5):
    return {
        "key_onset": {"f1": f1, "recall": recall},
        "top_voice": {"f1": f1},
        "selected_melody": {"f1": f1},
    }


def _row(candidate=0.5, original=0.5, recall=0.5, original_recall=0.5, added=0, non_drum_removed=0):
    return {
        "id": "demo", "split": "validation",
        "window_metrics": {window: _metrics(candidate, recall) for window in ("all", "front", "back")},
        "original_v3_window_metrics": {window: _metrics(original, original_recall) for window in ("all", "front", "back")},
        "v3_regression": {
            "selectedMelodyIdentical": True, "addedRenderedNoteCount": added,
            "removedRenderedNoteCount": 1, "filteredDrumCount": 2,
            "drumDerivedRemovedRenderedNoteCount": 1 - non_drum_removed,
            "nonDrumRemovedRenderedNoteCount": non_drum_removed,
        },
    }


class TestV3GoldenVerifier(unittest.TestCase):
    def test_accepts_drum_only_reduction_without_melody_regression(self):
        verdict = verify({"results": [_row()]})
        self.assertTrue(verdict["passed"])
        self.assertEqual(verdict["songCount"], 1)

    def test_rejects_new_output_but_reports_noise_sensitive_metric_drop_as_warning(self):
        verdict = verify({"results": [_row(candidate=.49, original=.5, added=1)]})
        reasons = {item["reason"] for item in verdict["failures"]}
        self.assertIn("non_drum_output_added", reasons)
        self.assertIn("top_voice_drop_after_drum_removal", {item["reason"] for item in verdict["warnings"]})

    def test_rejects_any_removed_non_drum_output(self):
        verdict = verify({"results": [_row(non_drum_removed=1)]})
        self.assertIn("non_drum_output_removed", {item["reason"] for item in verdict["failures"]})


if __name__ == "__main__":
    unittest.main()
