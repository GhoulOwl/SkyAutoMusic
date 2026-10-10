import unittest
from scripts.verify_fidelity import METRICS, verify
from transcription.benchmark import evaluate_15_key


class FidelityGateTests(unittest.TestCase):
    def report(self):
        return {'summary': {name: .5 for name in METRICS}, 'results': [{'id': 'a'}, {'id': 'b'}]}

    def test_missing_targeted_evidence_keeps_stable_default(self):
        self.assertFalse(verify(self.report(), self.report())['passed'])

    def test_improvement_and_nonregression_enable_candidate(self):
        verdict = verify(self.report(), self.report(), targeted={'passed': True})
        self.assertTrue(verdict['passed'])
        self.assertEqual(verdict['default_strategy'], 'candidate')

    def test_metric_drop_or_changed_sample_set_rejects_candidate(self):
        candidate = self.report()
        candidate['summary'][METRICS[-1]] = .49
        self.assertFalse(verify(self.report(), candidate, targeted={'passed': True})['passed'])

    def test_explicit_reference_melody_can_be_below_top_voice(self):
        reference = [{'time': 0, 'key': '1Key3'}, {'time': 0, 'key': '1Key10'},
                     {'time': 500, 'key': '1Key4'}, {'time': 500, 'key': '1Key11'}]
        lead = [reference[0], reference[2]]
        metrics = evaluate_15_key(reference, reference, selected_melody=lead, reference_melody=lead)
        self.assertEqual(metrics['selected_melody']['f1'], 1)
        self.assertEqual(metrics['reference_melody_kind'], 'explicit')
        candidate = self.report(); candidate['results'].pop()
        self.assertFalse(verify(self.report(), candidate, targeted={'passed': True})['passed'])
