"""Оценить сохранённую GRU-CNN на validation или test, измерить задержку."""
import argparse
import json
from pathlib import Path
import time

import torch

from gru_cnn.common import LABELS, ROOT, check_disjoint, fingerprint, metrics, read_rows, save_json
from gru_cnn.model import load_model, predict, tokenize
from telemetry import Telemetry


def evaluate(args):
    torch.set_num_threads(args.threads)
    metadata = json.loads((args.model_dir / 'training.json').read_text())
    data_dir = Path(metadata['data_dir'])
    references = {}
    for split in ('train', 'validation'):
        path = data_dir / f'{split}.jsonl'
        if fingerprint(path) != metadata['data_sha256'][split]:
            raise ValueError(f'Выборка {split} изменилась после обучения')
        references[split] = read_rows(path)
    dataset = data_dir / f'{args.split}.jsonl'
    rows = read_rows(dataset)
    check_disjoint(references['train'], rows)
    if args.split == 'test':
        check_disjoint(references['validation'], rows)
    model, vocab = load_model(args.model_dir / 'model.pt', args.fasttext_model)
    args.output.mkdir(parents=True, exist_ok=True)
    batches = [rows[i:i + args.batch_size] for i in range(0, len(rows), args.batch_size)]
    telemetry = Telemetry('gru-cnn', dataset, args.output, {
        'split': args.split, 'device': 'cpu', 'batch_size': args.batch_size,
        'warmup_batches': args.warmup, 'torch_threads': args.threads,
        'training_metadata': metadata,
        'latency_definition': 'End-to-end: tokenization, embedding (fastText vectors cached lazily), GRU, CNN, three heads and probabilities; excludes loading, warmup, metrics and file writes',
    })
    records, completed = [], False
    try:
        schedule = [('warmup', i, batches[i % len(batches)]) for i in range(args.warmup)]
        schedule += [('measured', i, batch) for i, batch in enumerate(batches)]
        for phase, index, batch in schedule:
            started = time.perf_counter()
            try:
                predictions = predict(model, vocab, [row['text'] for row in batch])
            except Exception as exc:
                telemetry.record(phase=phase, batch_index=index, status='error', examples=len(batch),
                                 latency_seconds=time.perf_counter() - started, error=str(exc))
                raise
            telemetry.record(phase=phase, batch_index=index, status='ok', examples=len(batch),
                             latency_seconds=time.perf_counter() - started,
                             ids=[row['id'] for row in batch])
            if phase == 'measured':
                records.extend({'id': row['id'], 'labels': row['labels'], **prediction}
                               for row, prediction in zip(batch, predictions))
        tokens = [token for row in rows for token in tokenize(row['text'], model.config['max_length'])]
        report = {'dataset': str(dataset), 'dataset_sha256': fingerprint(dataset),
                  'device': 'cpu', 'tasks': {},
                  'unknown_token_fraction': sum(token not in vocab for token in tokens) / max(1, len(tokens)),
                  'unknown_token_definition': 'Tokens absent from train vocabulary; fastText computes subword vectors instead of UNK',
                  'exact_match_accuracy': sum(row['labels'] == row['predictions'] for row in records) / len(records)}
        for task, labels in LABELS.items():
            score = metrics([row['labels'][task] for row in records],
                            [row['predictions'][task] for row in records], labels)
            report['tasks'][task] = score
            print(f"{args.split}/{task}: accuracy={score['accuracy']:.4f}, macro-F1={score['macro_f1']:.4f}")
        save_json(args.output / f'{args.split}_metrics.json', report)
        (args.output / f'{args.split}_predictions.jsonl').write_text(
            ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in records))
        completed = True
    finally:
        telemetry.finish(completed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('split', choices=('validation', 'test'), nargs='?', default='test')
    parser.add_argument('--model-dir', type=Path, default=ROOT / 'gru_cnn/weights-fasttext-en')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/gru-cnn-fasttext-en')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--fasttext-model', type=Path, help='Override location of pretrained binary; SHA-256 must match')
    args = parser.parse_args()
    if min(args.batch_size, args.threads) < 1 or args.warmup < 0:
        parser.error('batch-size и threads >= 1, warmup >= 0')
    evaluate(args)
