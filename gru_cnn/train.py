"""Обучить fastText → BiGRU → CNN на train; выбрать checkpoint по validation."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import random
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from gru_cnn.common import DATA, LABELS, ROOT, check_disjoint, fingerprint, metrics, read_rows, save_json
from gru_cnn.model import CONFIG, GRUCNN, FastTextEncoder, build_vocab, encode_inputs, predict


def train(args):
    if args.output.exists():
        raise ValueError('Каталог модели уже существует; укажите новый --output')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    train_rows = read_rows(args.data_dir / 'train.jsonl')
    validation = read_rows(args.data_dir / 'validation.jsonl')
    check_disjoint(train_rows, validation)
    for task, labels in LABELS.items():
        if {row['labels'][task] for row in train_rows} != set(labels):
            raise ValueError(f'В train отсутствуют классы {task}')
    vocab = build_vocab(train_rows, CONFIG['max_length'])
    config = dict(CONFIG)
    model = GRUCNN(config)
    model.fasttext_encoder = FastTextEncoder(args.fasttext_model, config['embedding_dim'])
    fasttext_metadata = {'path': str(args.fasttext_model.resolve()),
                         'sha256': model.fasttext_encoder.sha256, 'frozen': True}
    optimizer = torch.optim.NAdam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    best_score, best_epoch, history = -1, 0, []
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        indices = torch.randperm(len(train_rows)).tolist()
        total_loss = 0
        for start in range(0, len(indices), args.batch_size):
            batch = [train_rows[i] for i in indices[start:start + args.batch_size]]
            ids, lengths = encode_inputs(model, vocab, [row['text'] for row in batch])
            logits = model(ids, lengths)
            losses = [F.cross_entropy(logits[task], torch.tensor([
                labels.index(row['labels'][task]) for row in batch])) for task, labels in LABELS.items()]
            loss = torch.stack(losses).mean()
            if not torch.isfinite(loss):
                raise ValueError('Нечисловая функция потерь')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(batch)
        predictions = predict(model, vocab, [row['text'] for row in validation])
        reports = {task: metrics([row['labels'][task] for row in validation],
                                [row['predictions'][task] for row in predictions], labels)
                   for task, labels in LABELS.items()}
        score = sum(report['macro_f1'] for report in reports.values()) / len(LABELS)
        history.append({'epoch': epoch, 'train_loss': total_loss / len(train_rows),
                        'validation_mean_macro_f1': score, 'tasks': reports,
                        'seconds': time.perf_counter() - epoch_started})
        if score > best_score:
            best_score, best_epoch = score, epoch
            torch.save({'state_dict': model.state_dict(), 'vocab': vocab, 'config': config,
                        'fasttext': fasttext_metadata,
                        'labels': LABELS}, args.output / 'model.pt')
        print(f"Эпоха {epoch:02d}: loss={history[-1]['train_loss']:.4f}, "
              f"val mean-F1={score:.4f}, best={best_score:.4f} (эпоха {best_epoch}), "
              f"{history[-1]['seconds']:.2f} с", flush=True)
        save_json(args.output / 'history.json', history)
        if epoch - best_epoch >= args.patience:
            break
    metadata = {
        'architecture': config,
        'embedding': 'Frozen pretrained fastText with subword OOV vectors',
        'fasttext': fasttext_metadata,
        'labels': LABELS, 'vocabulary_size': len(vocab),
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'data_dir': str(args.data_dir.resolve()),
        'data_sha256': {split: fingerprint(args.data_dir / f'{split}.jsonl')
                        for split in ('train', 'validation')},
        'train_count': len(train_rows), 'validation_count': len(validation),
        'seed': args.seed, 'device': 'cpu', 'torch_threads': args.threads,
        'optimizer': 'NAdam', 'lr': args.lr, 'weight_decay': 1e-4,
        'batch_size': args.batch_size, 'max_epochs': args.epochs, 'patience': args.patience,
        'selection': 'Mean of three validation macro-F1 scores; earliest epoch wins ties; no refit',
        'best_epoch': best_epoch, 'best_validation_mean_macro_f1': best_score,
        'epochs_completed': len(history), 'training_seconds': time.perf_counter() - started,
        'versions': {name: importlib.metadata.version(name) for name in
                     ('torch', 'scikit-learn', 'fasttext')},
        'finished_at_utc': datetime.now(timezone.utc).isoformat(),
    }
    save_json(args.output / 'training.json', metadata)
    print(f'Модель сохранена в {args.output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DATA)
    parser.add_argument('--output', type=Path, default=ROOT / 'gru_cnn/weights-fasttext-en')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--patience', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--fasttext-model', type=Path, default=ROOT / 'data/embeddings/wiki.en.bin', help='Pretrained wiki.en.bin; vectors stay frozen')
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size, args.threads) < 1 or args.lr <= 0:
        parser.error('epochs, patience, batch-size, threads и lr должны быть положительными')
    train(args)
