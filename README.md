# Handwriting AI

Проект преобразует печатный текст в траекторию рукописного письма:
`[x, y, pen_up]`. Результат предназначен не только для рендера, но и для
будущего перьевого плоттера, поэтому модель предсказывает порядок движения и
подъёмы пера.

Одна обученная система воспроизводит один почерк. Переключение стилей и
многопользовательские embeddings намеренно не добавлены.

## Данные

В `dataset/` находятся:

- `jsons/` - исходные траектории `[x, y, pen_up]`;
- `texts/` - текстовые расшифровки;
- `all_trajectories.npz` - подготовленный набор для обучения.

Перед моделью абсолютные координаты преобразуются в `(dx, dy, pen_up)`,
сглаживаются и равномерно пересэмплируются внутри каждого штриха.

Проверка датасета:

```bash
.venv/bin/python main_training.py --stage audit
```

## Архитектура v7

Пайплайн обучается в следующем порядке:

```text
TrajectoryRecognizer
        |
        +--> CTC forced alignments
        |
LocalInkAutoencoder
        |
        +--> локальные ink-латенты
        |
ContentAlignedLatentFlow
        |
        +--> (dx, dy, pen_up)
```

### 1. TrajectoryRecognizer

CTC-распознаватель читает текст по реальной траектории. Он выполняет сразу
три задачи:

- измеряет CER;
- строит принудительное выравнивание символов и кадров;
- дает differentiable semantic loss для autoencoder и generator.

Recognizer всегда обучается первым.

### 2. LocalInkAutoencoder

Детерминированный autoencoder сжимает четыре точки в один latent-кадр.
Глобального Transformer-bottleneck больше нет: используются локальные
свертки с ограниченным receptive field. Поэтому latent около буквы описывает
локальный фрагмент пера, а не смешивает всю строку.

Помимо геометрических losses, реконструкция проходит через замороженный
recognizer. CTC loss заставляет autoencoder сохранять читаемый текст.

### 3. CTC forced alignment

Перед обучением generator recognizer строит Viterbi-выравнивание известной
расшифровки и реальной траектории. Результат кешируется в:

```text
runs/gtx1660/generator/forced_alignments.pt
```

Кеш содержит длительность каждого символа. При изменении датасета,
preprocessing или checkpoint-а recognizer он перестраивается автоматически.

Это фиксированная внешняя разметка. Generator больше не использует MAS и не
может получать награду за собственное ошибочное выравнивание.

### 4. ContentAlignedLatentFlow

Text encoder и `DurationPredictor` разворачивают символы в latent-кадры.
Каждый кадр получает:

- embedding соответствующего символа;
- относительную позицию внутри символа;
- длительность символа;
- время rectified flow.

Flow-блоки локальные и имеют линейную сложность по длине, что важно для
GTX 1660 после уменьшения `downsample_factor` с 8 до 4.

Generator оптимизируется по:

- velocity flow matching;
- ошибке длительностей из CTC-разметки;
- latent endpoint loss;
- `xy`, cumulative path, `pen_up` и curvature после декодирования;
- CTC semantic loss по предсказанной конечной траектории.

`best.pt` выбирается прежде всего по CER реально сгенерированных validation-
примеров, а не только по внутреннему flow loss.

## Установка и тесты

Проект рассчитан на Python 3.11+ и PyTorch.

```bash
.venv/bin/python -m unittest discover -s tests
.venv/bin/python main_training.py --profile cpu_smoke --stage audit
```

Полный однопроходный smoke-тест:

```bash
.venv/bin/python main_training.py --profile cpu_smoke --stage all
.venv/bin/python main_evaluate.py --profile cpu_smoke --stage all --count 2
```

`cpu_smoke` проверяет работоспособность кода, но не качество почерка.

## Обучение

### Шаг 1. Recognizer

Обычный запуск `main_training.py` теперь обучает именно recognizer:

```bash
.venv/bin/python main_training.py
```

То же самое явно:

```bash
.venv/bin/python main_training.py --stage recognizer
.venv/bin/python main_evaluate.py --stage recognizer
```

Главная метрика - `val_cer`. На текущем наборе хороший ориентир:

- `< 0.10` - можно переходить дальше;
- `0.10-0.20` - forced alignment возможен, но стоит проверить ошибки;
- `> 0.20` - сначала улучшить recognizer.

### Шаг 2. Local autoencoder

```bash
.venv/bin/python main_training.py --stage autoencoder
.venv/bin/python main_evaluate.py --stage autoencoder --split train --count 8
.venv/bin/python main_evaluate.py --stage autoencoder --split val --count 8
```

Реконструкции должны сохранять буквы, соединения и подъемы пера. Проверять
только низкий `xy_loss` недостаточно: важны визуальная читаемость и
`val_cer`.

### Шаг 3. Generator

```bash
.venv/bin/python main_training.py --stage generator
```

В начале запуска будут рассчитаны latent statistics и CTC forced alignments.
Повторный запуск использует кеш, если его входные данные не изменились.

Проверка:

```bash
.venv/bin/python main_evaluate.py --stage generator --split train --count 8 --generator-selection train_best
.venv/bin/python main_evaluate.py --stage generator --split val --count 8
.venv/bin/python main_evaluate.py --stage generator --split val --count 8 --use-predicted-length
```

Сначала generator должен писать train-примеры, затем validation. Основная
метрика в логе - `val_cer`. Дополнительно контролируются:

- `velocity`;
- `duration`;
- `endpoint_latent`;
- `decoded_xy`, `decoded_path`, `decoded_pen`;
- `semantic`.

Падение внутренних losses без улучшения `val_cer` больше не считается
успешным обучением.

### Все этапы

```bash
.venv/bin/python main_training.py --stage all
```

Порядок будет выбран автоматически:

```text
recognizer -> autoencoder -> forced alignments -> generator
```

Чтобы пропустить уже готовые checkpoint-ы текущей версии:

```bash
.venv/bin/python main_training.py --stage all --skip-existing
```

## Несовместимость старых checkpoint-ов

Архитектура v7 требует переобучить autoencoder и generator:

```bash
.venv/bin/python main_training.py --stage autoencoder
.venv/bin/python main_training.py --stage generator
```

Старый recognizer можно использовать, если он хорошо распознает validation-
набор. Старые `InkAutoencoder` и `AlignedLatentFlow v6` несовместимы с новым
обучением.

Новые типы checkpoint-ов:

```text
model_type = local_content_autoencoder
model_type = content_aligned_latent_flow
generator_training_version = 7
```

Скрипт генерации не будет молча загружать старый generator.

## Генерация

```bash
.venv/bin/python main_generate.py "Привет, это рукописная строка."
```

Результаты сохраняются в `outputs/generated/`:

- `.json` - траектория для плоттера;
- `.png` - визуальная проверка.

Примеры параметров:

```bash
.venv/bin/python main_generate.py "Проверка." --temperature 0.8
.venv/bin/python main_generate.py "Другой вариант." --temperature 1.1
.venv/bin/python main_generate.py "Диагностика длины." --latent-length 320
.venv/bin/python main_generate.py "Train checkpoint." --generator-selection train_best
.venv/bin/python main_generate.py "Именованный файл." --name sample
```

Flow обучается от Gaussian noise, поэтому нормальный диапазон температуры -
примерно `0.7-1.1`. `temperature=0` полезен только как диагностическая
условная средняя и может давать неестественный результат.

`--steps` задает число шагов интегрирования. Для быстрой проверки можно
использовать 4-8, для итоговой генерации профиль GTX 1660 использует 32.

## Настройки GTX 1660

Основной профиль: `configs/gtx1660.toml`.

В нем уже включены:

- `batch_size = 2`;
- AMP;
- gradient accumulation;
- `persistent_workers`;
- validation раз в пять эпох;
- локальный flow без квадратичного attention по всей траектории;
- semantic CTC loss раз в четыре шага;
- CER-оценка на ограниченном числе validation-примеров.

Если не хватает VRAM:

1. поставить `data.batch_size = 1`;
2. увеличить `generator.grad_accum_steps`;
3. уменьшить `data.max_points`;
4. уменьшить `generator.hidden_dim`;
5. уменьшить `generator.layers`.

Если пауза между эпохами слишком длинная:

1. увеличить `eval_every`;
2. уменьшить `generator.eval_samples`;
3. уменьшить `generator.eval_flow_steps`;
4. увеличить `semantic_every`;
5. реже сохранять `epoch_*.pt` через `checkpoint_every`.

Эти параметры влияют на скорость контроля, но не отключают основные
геометрические losses.

## Диагностика

Минимальный порядок:

```bash
.venv/bin/python main_training.py --stage audit
.venv/bin/python main_evaluate.py --stage recognizer
.venv/bin/python main_evaluate.py --stage autoencoder --split val --count 8
.venv/bin/python main_evaluate.py --stage generator --split train --count 8 --generator-selection train_best
.venv/bin/python main_evaluate.py --stage generator --split val --count 8
```

Интерпретация:

1. Плохой CER recognizer означает ненадежные границы символов.
2. Хороший recognizer, но нечитаемый autoencoder означает потерю информации
   в latent-пространстве.
3. Хороший autoencoder, но плохой train generator означает проблему flow,
   длительностей или весов semantic/geometric losses.
4. Хороший train и плохой val означают недостаток покрытия датасета.

## Основные файлы

- `main_training.py` - обучение этапов;
- `main_generate.py` - генерация JSON и PNG;
- `main_evaluate.py` - визуальная диагностика;
- `src/handwriting_ai/alignment.py` - CTC forced alignment и кеш;
- `src/handwriting_ai/metrics.py` - CER и CTC decode;
- `src/handwriting_ai/models/local_autoencoder.py` - локальный autoencoder;
- `src/handwriting_ai/models/content_flow.py` - generator v7;
- `tests/` - unit и integration-тесты.

## Низкоуровневый CLI

```bash
PYTHONPATH=src .venv/bin/python -m handwriting_ai audit-data --config configs/gtx1660.toml
PYTHONPATH=src .venv/bin/python -m handwriting_ai render --json-path outputs/generated/sample.json --out outputs/generated/sample.png
```

## Передача эпох по локальной сети

`main_sync.py` передает только checkpoint-ы из проектной папки `runs/`.
Датасет и исходный код он не синхронизирует.

На принимающем компьютере:

```bash
python main_sync.py serve --token my_secret
```

На компьютере с видеокартой:

```bash
python main_sync.py watch --remote http://192.168.1.20:8765 --token my_secret
```

Разовая отправка:

```bash
python main_sync.py push --remote http://192.168.1.20:8765 --token my_secret
```

Только `best.pt`, `train_best.pt`, `last.pt` и `config.json`:

```bash
python main_sync.py watch --remote http://192.168.1.20:8765 --token my_secret --best-last-only
```

Явная папка эпох:

```bash
python main_sync.py serve --epochs-dir /Users/frimo/Documents/PycharmProjects/Handwriting/runs --token my_secret
python main_sync.py watch --epochs-dir /Users/frimo/Documents/PycharmProjects/Handwriting/runs --remote http://192.168.1.20:8765 --token my_secret
```

В логе:

- `UP` - файл передан;
- `SKIP` - такой же файл уже есть;
- `RECV` - файл принят сервером.

Для каждого файла и полного прохода выводятся размер, время и скорость.

Оба компьютера обычно могут работать через локальный IP одного роутера даже
при включенном VPN. Если соединения нет, нужно разрешить local network/LAN в
настройках VPN и проверить firewall.
