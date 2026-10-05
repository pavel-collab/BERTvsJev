"""Общие средства записи телеметрии существующих скриптов."""
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics


def latency_summary(values, examples=None):
    if not values:
        return None
    ordered = sorted(values)
    def percentile(q):
        pos = (len(ordered) - 1) * q
        low, high = math.floor(pos), math.ceil(pos)
        return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)
    total = sum(values)
    return {'count': len(values), 'total_seconds': total,
            'mean_seconds': statistics.mean(values),
            'stdev_seconds': statistics.stdev(values) if len(values) > 1 else 0,
            'min_seconds': min(values), 'max_seconds': max(values),
            **{f'p{int(q * 100)}_seconds': percentile(q) for q in (.5, .95, .99)},
            'examples_per_second': (examples if examples is not None else len(values)) / total if total else None}


class Telemetry:
    def __init__(self, backend, dataset, parent, metadata):
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        self.path = Path(parent) / 'telemetry' / f'{backend}-{stamp}'
        self.path.mkdir(parents=True, exist_ok=False)
        self.records = []
        self.metadata = {'backend': backend, 'started_at_utc': datetime.now(timezone.utc).isoformat(),
                         'dataset': str(dataset), 'dataset_sha256': hashlib.sha256(Path(dataset).read_bytes()).hexdigest(),
                         'platform': platform.platform(), 'machine': platform.machine(),
                         'python': platform.python_version(), **metadata}
        self.save('metadata.json', self.metadata)
        self.handle = (self.path / 'measurements.jsonl').open('x')

    def save(self, name, value):
        (self.path / name).write_text(json.dumps(value, ensure_ascii=False, indent=2))

    def record(self, **value):
        self.handle.write(json.dumps(value, ensure_ascii=False) + '\n')
        self.handle.flush()
        self.records.append(value)

    def finish(self, completed, extra=None):
        self.handle.close()
        measured = [r for r in self.records if r['phase'] == 'measured']
        successful = [r for r in measured if r['status'] == 'ok']
        summary = {'completed': completed, 'finished_at_utc': datetime.now(timezone.utc).isoformat(),
                   'attempts': len(measured), 'errors': len(measured) - len(successful),
                   'error_rate': (len(measured) - len(successful)) / len(measured) if measured else None,
                   'latency': latency_summary([r['latency_seconds'] for r in successful], sum(r.get('examples', 1) for r in successful)),
                   **(extra or {})}
        self.save('summary.json', summary)
        self.save('metadata.json', self.metadata)
        fields = sorted({key for record in self.records for key in record})
        with (self.path / 'measurements.csv').open('x', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for record in self.records:
                writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value for key, value in record.items()})
        print(f'Телеметрия: {self.path}')
