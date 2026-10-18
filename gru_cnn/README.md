# fastText → GRU → CNN

Классификатор темы, срочности и тональности обращения с фиксированными
300-мерными предобученными векторами `wiki.en.bin`. Архитектура адаптирована
из `page-stream-segmentation`, класс `TextPageEncoder` в `app/pss/model.py`.
Лицензия исходного кода сохранена в `LICENSE.source`.

## Архитектура

| Этап | Операция | Размер выхода |
|---|---|---|
| Токенизация | Слова и группы пунктуации, lowercase, замена URL, до 150 токенов | B × L |
| fastText | Фиксированные векторы, включая подсловные представления незнакомых слов | B × L × 300 |
| Контекст | Независимые прямой и обратный GRU, по 128 признаков | B × L × 256 |
| CNN | Три Conv1d, ядра 3/4/5, по 200 каналов, SAME padding | 3 × B × 200 × L |
| Pooling | LeakyReLU(0.3), masked global max-pooling, Dropout(0.5) | B × 600 |
| Представление | Linear(600, 128), LeakyReLU(0.3) | B × 128 |
| Три головы | Отдельные Linear(128, 4/2/3), softmax при оценке | topic / urgent / sentiment |

Обучаемых параметров — 1 023 329. fastText не входит в оптимизатор.
Input dropout 0.5 имеет общую маску по времени и независимые маски
для двух направлений. Длинные тексты сокращаются до первых и последних
75 токенов. Padding исключён из GRU через packed sequences и замаскирован
перед max-pooling. Пустой текст представлен одним нулевым вектором.

Один энкодер обслуживает три головы. В исходном проекте голова бинарная
и определяет начало документа; здесь каждое обращение независимо.
Контекст предыдущей страницы не используется.

Словарь train используется только для статистики. Незнакомые слова
обрабатываются `get_word_vector`, без замены на общий UNK. Векторы кешируются
по токенам; кеш не использует метки и не обучается на validation/test.

## Векторы и запуск

Источник — [официальные Wikipedia-векторы fastText](https://fasttext.cc/docs/en/pretrained-vectors.html),
лицензия CC BY-SA 3.0. Извлеките `wiki.en.bin` из
[архива](https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip)
в `data/embeddings/`. Архив около 10,36 ГБ, бинарный файл — 8,49 ГБ;
при обычной загрузке и распаковке нужно место для обоих.
Бинарная модель нужна для подсловных представлений; `.vec` не является
равноценной заменой. Используется fastText, а не библиотека FastEmbed.

Из корня проекта:

```sh
uv sync --locked
.venv/bin/python -m gru_cnn.train
.venv/bin/python -m gru_cnn.test validation
.venv/bin/python -m gru_cnn.test test --batch-size 1
```

Модель уже обучена. Для повторного обучения укажите новый каталог:

```sh
.venv/bin/python -m gru_cnn.train --output gru_cnn/weights-repeat
.venv/bin/python -m gru_cnn.test test --model-dir gru_cnn/weights-repeat --output results/gru-cnn-repeat
```

По умолчанию используются `data/embeddings/wiki.en.bin`,
`gru_cnn/weights-fasttext-en/` и `results/gru-cnn-fasttext-en/`.
Путь к бинарному fastText можно изменить через `--fasttext-model`.
Checkpoint содержит параметры сети и путь/SHA-256 внешней модели;
бинарный fastText в checkpoint не включён и нужен при загрузке.
При переносе файла контрольная сумма должна совпадать.

```python
from gru_cnn.model import load_model, predict
model, vocab = load_model('gru_cnn/weights-fasttext-en/model.pt')
print(predict(model, vocab, ['Our checkout is down. Please help immediately!']))
```

## Методика

Train/validation/test: 144/48/48 синтетических англоязычных обращений.
Пересечения `id` и `group_id` проверяются. Обучение не читает test;
оценка проверяет SHA-256 train и validation. Проверки и метрики находятся
в `gru_cnn/common.py`.

CPU, 2 потока PyTorch, seed=42, детерминированные операции, NAdam,
lr=0.001, weight_decay=0.0001, batch_size=16, clipping градиентов 1.0.
Loss — среднее трёх cross-entropy. Максимум 60 эпох, early stopping после
12 эпох без улучшения среднего validation macro-F1; при равенстве остаётся
ранний checkpoint. Настройки по test не подбирались.

## Результаты 18 октября 2026

Выбрана эпоха 30 из 42; средний validation macro-F1 — 0,9714.
Цикл обучения занял 35,47 с без загрузки и проверки внешней модели.

| Метрика test | Jev | BERT | fastText → GRU → CNN |
|---|---:|---:|---:|
| Тема, accuracy | 97,92% | 97,92% | 93,75% |
| Тема, macro-F1 | 0,9791 | 0,9791 | 0,9365 |
| Срочность, accuracy | 100,00% | 100,00% | 95,83% |
| Срочность, macro-F1 | 1,0000 | 1,0000 | 0,9583 |
| Тональность, accuracy | 85,42% | 97,92% | 91,67% |
| Тональность, macro-F1 | 0,8562 | 0,9791 | 0,9169 |
| Среднее macro-F1 | 0,9451 | 0,9861 | 0,9372 |
| Все три метки верны | 40/48 (83,33%) | 46/48 (95,83%) | 40/48 (83,33%) |

ID и разметка совпадают между тремя моделями. Jev и BERT повторно не запускались.
Это один запуск на малом синтетическом корпусе; тест ранее анализировался.
Для независимого подтверждения нужен новый отложенный набор и несколько seed.

| Latency трёх меток на обращение | Jev API | BERT (MPS) | fastText → GRU → CNN (CPU) |
|---|---:|---:|---:|
| Среднее, мс | 388,24 | 37,85 | 1,34 |
| p50, мс | 358,26 | 28,74 | 1,08 |
| p95, мс | 526,19 | 94,55 | 2,98 |

По 48 обращений, batch_size=1. Локальные модели измерены после прогрева,
без загрузки весов; fastText использует ленивый кеш векторов.
Для BERT суммируются отдельные замеры трёх классификаторов.
У Jev включены сеть и проверка ответа SDK. Запуски выполнены в разные дни;
это короткие замеры, не нагрузочный тест.

## Артефакты

- `weights-fasttext-en/model.pt`: лучший checkpoint и словарь train для статистики.
- `weights-fasttext-en/training.json`: настройки, версии, SHA-256 и выбранная эпоха.
- `weights-fasttext-en/history.json`: динамика обучения.
- `../results/gru-cnn-fasttext-en/{validation,test}_metrics.json`: метрики.
- `../results/gru-cnn-fasttext-en/{validation,test}_predictions.jsonl`: предсказания.
- `../results/gru-cnn-fasttext-en/telemetry/`: замеры задержки.
- `../data/embeddings/wiki.en.bin`: внешняя предобученная модель.
- `../data/embeddings/wiki.en.json`: источник и SHA-256 внешней модели.

`unknown_token_fraction` — доля токенов вне словаря train, а не доля
потерянных представлений. Размер внешнего fastText не входит в размер checkpoint.
