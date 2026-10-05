import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import urllib.error
import http.client
from unittest.mock import patch

import jev
from telemetry import latency_summary


class TelemetryTests(unittest.TestCase):
    def test_statistics_and_cost(self):
        self.assertEqual(latency_summary([1, 2, 3])['p50_seconds'], 2)
        self.assertAlmostEqual(jev.usage_cost({'usage': {'input_tokens': 1000}}, .042, 0)[1], .000042)
        self.assertIsNone(jev.usage_cost({'usage': {}}, .042, 0)[1])

    def test_request_failures(self):
        failures = [
            (urllib.error.HTTPError('https://api.typesafe.ai', 429, 'limited', {'Retry-After': '12'}, io.BytesIO()), 'http'),
            (urllib.error.URLError('DNS failed'), 'network'),
            (urllib.error.URLError(TimeoutError()), 'timeout'),
            (TimeoutError(), 'timeout'),
            (http.client.IncompleteRead(b'partial'), 'connection'),
        ]
        for error, kind in failures:
            with self.subTest(kind=kind), patch('urllib.request.urlopen', side_effect=error):
                with self.assertRaises(jev.JevRequestError) as caught:
                    jev.call_jev(jev.request_payload('text', 'model'), 'secret', 1)
                self.assertEqual(caught.exception.kind, kind)
                self.assertNotIn('secret', str(caught.exception))
                if kind == 'http':
                    self.assertEqual(caught.exception.http_status, 429)
                    self.assertEqual(caught.exception.retry_after, '12')

    def test_response_validation(self):
        for raw, kind in [(b'not json', 'invalid_json'), (b'[]', 'invalid_response'),
                          (b'{"answers": {}}', 'invalid_response'),
                          (b'{"answers": {"topic": {}, "urgent": {}, "sentiment": {}}, "usage": []}', 'invalid_response')]:
            with self.subTest(raw=raw), patch('urllib.request.urlopen', return_value=io.BytesIO(raw)):
                with self.assertRaises(jev.JevRequestError) as caught:
                    jev.call_jev(jev.request_payload('text', 'model'), 'secret', 1)
                self.assertEqual(caught.exception.kind, kind)

    def test_partial_run_is_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / 'input.jsonl'
            dataset.write_text('\n'.join(json.dumps({'id': str(i), 'text': 'hello'}) for i in range(2)))
            response = {'answers': {task: {} for task in jev.QUESTIONS}, 'usage': {'input_tokens': 1000}}
            argv = ['jev.py', 'run', '--input', str(dataset), '--output', str(root / 'out.jsonl')]
            with patch.object(sys, 'argv', argv), patch.object(jev, 'load_env'), patch.dict('os.environ', {'TYPESAFE_API_KEY': 'secret'}), patch.object(jev, 'call_jev', side_effect=[(response, .1), jev.JevRequestError('timeout', 'timeout')]):
                with self.assertRaises(jev.JevRequestError):
                    jev.main()
            summary = json.loads(next(root.glob('telemetry/*/summary.json')).read_text())
            self.assertFalse(summary['completed'])
            self.assertEqual(summary['attempts'], 2)
            self.assertEqual(summary['errors'], 1)
            self.assertIsNone(summary['estimated_total_cost_usd'])
            self.assertEqual(len((root / 'out.jsonl').read_text().splitlines()), 1)


if __name__ == '__main__':
    unittest.main()
