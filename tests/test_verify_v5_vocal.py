import unittest

from scripts.verify_v5_vocal import verify


def _report(onset=0.5, selected=0.5, validation_onset=0.5, validation_selected=0.5, recall=None):
    recall = selected if recall is None else recall
    return {
        "summary": {
            "median_onset_f1": onset,
            "median_selected_melody_f1": selected,
            "validation_median_onset_f1": validation_onset,
            "validation_median_selected_melody_f1": validation_selected,
        },
        "results": [{
            "id": "song", "metrics": {"selected_melody": {"recall": recall}},
        }],
    }


class TestVerifyV5Vocal(unittest.TestCase):
    def test_accepts_vocal_recall_and_metric_improvements(self):
        verdict = verify(_report(), _report(onset=.51, selected=.52, validation_onset=.5, validation_selected=.5))
        self.assertTrue(verdict["passed"])

    def test_allows_bounded_metric_drop(self):
        verdict = verify(_report(), _report(onset=.495, selected=.495, validation_onset=.495, validation_selected=.495, recall=.5))
        self.assertTrue(verdict["passed"])

    def test_rejects_selected_melody_recall_drop(self):
        verdict = verify(_report(), _report(selected=.49))
        self.assertIn("selected_melody_recall_drop", {item["reason"] for item in verdict["failures"]})


if __name__ == "__main__":
    unittest.main()
