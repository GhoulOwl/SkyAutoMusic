import unittest
from scripts.verify_fidelity import METRICS, verify


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
        candidate = self.report(); candidate['results'].pop()
        self.assertFalse(verify(self.report(), candidate, targeted={'passed': True})['passed'])
