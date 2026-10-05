import json
from pathlib import Path
import tempfile
import unittest

from jev_evaluate import auc_metrics, convert, read_results


class EvaluationTests(unittest.TestCase):
    def test_conversion_uses_distribution_not_rounded_score_or_confidence(self):
        record = {"response": {"answers": {
            "topic": {"choice": "billing", "confidence": 0.2,
                      "probabilities": {"billing": 0.7, "technical": 0.1, "sales": 0.1, "other": 0.1}},
            "urgent": {"noul": 0.4},
            "sentiment": {"score": 0.9, "confidence": 0.1,
                          "probabilities": {"0": 0.55, "1": 0, "2": 0.45}},
        }}}
        predictions, probs = convert(record, 0.5)
        self.assertEqual(predictions, {"topic": "billing", "urgent": 0, "sentiment": 0})
        self.assertEqual(probs["urgent"], {"0": 0.6, "1": 0.4})
        self.assertEqual(convert(record, 0.3)[0]["urgent"], 1)

    def test_auc_uses_continuous_scores(self):
        result = auc_metrics([0, 1, 0, 1], [[0.9, 0.1], [0.7, 0.3], [0.8, 0.2], [0.6, 0.4]], [0, 1])
        self.assertEqual(result["roc_auc"], 1)
        self.assertIsNone(auc_metrics([0, 0], [[0.8, 0.2], [0.7, 0.3]], [0, 1])["roc_auc"])

    def test_multiclass_auc(self):
        result = auc_metrics([0, 1, 2], [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]], [0, 1, 2])
        self.assertEqual(result["roc_auc"], 1)

    def test_json_jsonl_and_duplicates(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "answers.json"
            records = [{"id": "a"}, {"id": "b"}]
            for text in (json.dumps(records), "\n".join(map(json.dumps, records))):
                path.write_text(text)
                self.assertEqual(set(read_results(path)), {"a", "b"})
            path.write_text(json.dumps([records[0], records[0]]))
            with self.assertRaises(ValueError):
                read_results(path)


if __name__ == "__main__":
    unittest.main()
