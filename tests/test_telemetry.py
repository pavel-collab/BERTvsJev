import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import asyncio
import httpx2
from unittest.mock import patch, AsyncMock

import jev
from telemetry import latency_summary


class TelemetryTests(unittest.TestCase):
    def test_statistics_and_cost(self):
        self.assertEqual(latency_summary([1, 2, 3])['p50_seconds'], 2)
        self.assertAlmostEqual(jev.usage_cost({'usage': {'input_tokens': 1000}}, .042, 0)[1], .000042)
        self.assertIsNone(jev.usage_cost({'usage': {}}, .042, 0)[1])

    def test_request_failures(self):
        failures = [
            (httpx2.Response(429, headers={'Retry-After': '12'}, text='secret'), 'http'),
            (httpx2.ConnectError('secret'), 'network'),
            (httpx2.ReadTimeout('secret'), 'timeout'),
            (httpx2.ReadError('secret'), 'network'),
        ]
        for error, kind in failures:
            attempts = []
            def respond(request):
                attempts.append(request)
                if isinstance(error, Exception):
                    raise error
                return error
            async def check():
                async with jev.AsyncTypeSafeClient(api_key='secret', timeout=1,
                        retry=jev.RetryPolicy(max_retries=0),
                        transport=httpx2.MockTransport(respond)) as client:
                    with self.assertRaises(jev.JevRequestError) as caught:
                        await jev.call_jev(jev.request_payload('text', 'model'), client)
                    self.assertEqual(caught.exception.kind, kind)
                    self.assertNotIn('secret', str(caught.exception))
                    if kind == 'http':
                        self.assertEqual(caught.exception.http_status, 429)
                        self.assertEqual(caught.exception.retry_after, '12')
            with self.subTest(kind=kind):
                asyncio.run(check())
                self.assertEqual(len(attempts), 1)

    def test_response_validation(self):
        for raw in [b'not json', b'[]', b'{"answers": {}}',
                    b'{"model": "jev", "answers": {}, "usage": {}}',
                    b'{"answers": {"topic": {}, "urgent": {}, "sentiment": {}}, "usage": []}']:
            async def check():
                async with jev.AsyncTypeSafeClient(api_key='secret',
                        retry=jev.RetryPolicy(max_retries=0),
                        transport=httpx2.MockTransport(lambda request: httpx2.Response(200, content=raw))) as client:
                    with self.assertRaises(jev.JevRequestError) as caught:
                        await jev.call_jev(jev.request_payload('text', 'model'), client)
                    self.assertEqual(caught.exception.kind, 'invalid_response')
            with self.subTest(raw=raw):
                asyncio.run(check())

    def test_sdk_request_and_saved_response(self):
        response = {'model': 'jev-latest', 'usage': {'input_tokens': 100, 'output_tokens': 0},
                    'answers': {
                        'topic': {'type': 'choice', 'choice': 'billing', 'confidence': 0.7,
                                  'probabilities': {'billing': 0.7, 'technical': 0.1, 'sales': 0.1, 'other': 0.1}},
                        'urgent': {'type': 'noul', 'noul': 0.4},
                        'sentiment': {'type': 'score', 'score': 0.9, 'confidence': 0.55,
                                      'legend': {'0': 'negative', '1': 'neutral', '2': 'positive'},
                                      'probabilities': {'0': 0.55, '1': 0.0, '2': 0.45}},
                    }}
        requests = []
        def respond(request):
            requests.append(request)
            return httpx2.Response(200, json=response)
        async def check():
            async with jev.AsyncTypeSafeClient(api_key='secret',
                    retry=jev.RetryPolicy(max_retries=0),
                    transport=httpx2.MockTransport(respond)) as client:
                for text in ['hello', 'world']:
                    payload = jev.request_payload(text, 'jev-latest')
                    result, elapsed = await jev.call_jev(payload, client)
                    self.assertEqual(json.loads(requests[-1].content), payload)
                    self.assertEqual(result, response)
                    from jev_evaluate import convert
                    self.assertEqual(convert({'response': result}, 0.5)[0],
                                     {'topic': 'billing', 'urgent': 0, 'sentiment': 0})
                    self.assertGreaterEqual(elapsed, 0)
                    self.assertIsInstance(next(iter(result['answers']['sentiment']['probabilities'])), str)
            self.assertTrue(client._http_client.is_closed)
        asyncio.run(check())
        self.assertEqual(len(requests), 2)

    def test_partial_run_is_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / 'input.jsonl'
            dataset.write_text('\n'.join(json.dumps({'id': str(i), 'text': 'hello'}) for i in range(2)))
            response = {'answers': {task: {} for task in jev.QUESTIONS}, 'usage': {'input_tokens': 1000}}
            argv = ['jev.py', 'run', '--input', str(dataset), '--output', str(root / 'out.jsonl')]
            with patch.object(sys, 'argv', argv), patch.object(jev, 'load_env'), patch.dict('os.environ', {'TYPESAFE_API_KEY': 'secret'}), patch.object(jev, 'call_jev', new_callable=AsyncMock, side_effect=[(response, .1), jev.JevRequestError('timeout', 'timeout')]):
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
