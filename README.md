# Handwriting AI

Проект генерирует рукописные траектории пера из печатного текста. Цель не в том,
чтобы получить картинку почерка, а в том, чтобы получить порядок движения пера,
подъёмы пера и координаты, пригодные для будущего плоттера.

Датасет уже лежит в `dataset/`:

- `dataset/jsons/` - исходные траектории `[x, y, pen_up]`;
- `dataset/texts/` - текстовые расшифровки;
- `dataset/all_trajectories.npz` - объединённый файл для обучения.

## Архитектура

Пайплайн состоит из трёх моделей.

1. `InkAutoencoder`

   Сжимает длинную последовательность `(dx, dy, pen_up)` в короткую latent-
   последовательность и восстанавливает её обратно. Это главный первый этап:
   если автоэнкодер плохо восстанавливает реальные строки, генератор тоже не
   сможет писать хорошо.

2. `AlignedLatentFlow`

   Генерирует latent-последовательность автоэнкодера по тексту. Во время
   обучения `Monotonic Alignment Search` автоматически определяет, сколько
   latent-кадров относится к каждому символу. `DurationPredictor` учится
   воспроизводить эти длительности, а conditional rectified flow с
   Conformer-блоками генерирует саму последовательность. Это устраняет
   принудительное равномерное выравнивание символов и усреднение траекторий в
   прямую линию.

3. `TrajectoryRecognizer`

   CTC-распознаватель траекторий. Он нужен для контроля читаемости, будущего
   расчёта CER и rerank нескольких сгенерированных вариантов.

## Основные файлы

- `main_training.py` - удобный запуск обучения.
- `main_generate.py` - удобная генерация тестовых примеров.
- `main_evaluate.py` - визуальные проверки между этапами обучения.
- `configs/gtx1660.toml` - основной профиль под Ryzen 5 2600, GTX 1660 и 16 GB RAM.
- `configs/cpu_smoke.toml` - очень маленький профиль для быстрой проверки кода.
- `src/handwriting_ai/` - основной пакет.
- `tests/` - тесты на preprocessing, модели и inference.

## Быстрая проверка

Перед долгим обучением стоит проверить, что код и датасет читаются:

```bash
.venv/bin/python main_training.py --profile cpu_smoke --stage audit
.venv/bin/python -m unittest discover -s tests
```

Можно прогнать весь мини-пайплайн на CPU:

```bash
.venv/bin/python main_training.py --profile cpu_smoke --stage all
.venv/bin/python main_generate.py --profile cpu_smoke "Тест." --latent-length 24 --steps 2 --temperature 0
.venv/bin/python main_evaluate.py --profile cpu_smoke --stage all --count 2
```

После этого в `outputs/generated/` появятся `.json` и `.png`.

## Обучение

Первым всегда обучается автоэнкодер:

```bash
.venv/bin/python main_training.py --stage autoencoder
```

После этого обязательно проверить реконструкции:

```bash
.venv/bin/python main_evaluate.py --stage autoencoder --split val --count 8
```

Открой картинки в `outputs/eval/val/autoencoder/`. Реконструкция должна быть
похожа на оригинал почти до уровня отдельных букв. Если она превращает буквы в
палочки или ломает соединения, генератор дальше не спасёт ситуацию.

Когда появится `runs/gtx1660/autoencoder/best.pt`, можно обучать генератор:

```bash
.venv/bin/python main_training.py --stage generator
```

После генератора проверить хотя бы train и val:

```bash
.venv/bin/python main_evaluate.py --stage generator --split train --count 8 --generator-selection train_best
.venv/bin/python main_evaluate.py --stage generator --split val --count 8
```

Рендеры будут разнесены по папкам `outputs/eval/train/generator/` и
`outputs/eval/val/generator/`, чтобы train и val не смешивались.
`train_best.pt` нужен для диагностики: если он не пишет train-примеры, то
генератор ещё не научился запоминать даже обучающую выборку. Обычный `best.pt`
по-прежнему выбирается по validation loss.

По умолчанию `main_evaluate.py` для генератора берёт latent-длину из реального
примера. Это упрощает диагностику: если даже при правильной длине строка
нечитаемая, значит проблема находится в alignment или flow, а не в
`DurationPredictor`.
Позже можно проверить предсказание длины отдельно:

```bash
.venv/bin/python main_evaluate.py --stage generator --split val --count 8 --use-predicted-length
```

Распознаватель обучается отдельно:

```bash
.venv/bin/python main_training.py --stage recognizer
```

Проверить его метрики:

```bash
.venv/bin/python main_evaluate.py --stage recognizer
```

Если нужно запустить всё подряд:

```bash
.venv/bin/python main_training.py --stage all
```

Если обучение уже частично было пройдено и нужно не трогать готовые
чекпойнты:

```bash
.venv/bin/python main_training.py --stage all --skip-existing
```

## Генерация

После обучения автоэнкодера и генератора:

```bash
.venv/bin/python main_generate.py "Привет, это рукописная строка."
```

По умолчанию скрипт берёт:

- `runs/gtx1660/autoencoder/best.pt`;
- `runs/gtx1660/generator/best.pt`;
- настройки из `configs/gtx1660.toml`;
- сохраняет результат в `outputs/generated/`.

Полезные параметры:

```bash
.venv/bin/python main_generate.py "Проверка температуры." --temperature 0.7
.venv/bin/python main_generate.py "Более смелый вариант." --temperature 1.1
.venv/bin/python main_generate.py "Фиксированная длина." --latent-length 320
.venv/bin/python main_generate.py "Train-best диагностика." --generator-selection train_best
.venv/bin/python main_generate.py "Именованный файл." --name my_sample
```

Файл `.json` содержит траекторию для плоттера, `.png` нужен только для
быстрого визуального контроля.

`--steps` задаёт число шагов интегрирования rectified flow. Для обычной
генерации используется значение из конфига, а для быстрой проверки можно
поставить `--steps 4`. `--latent-length` вручную задаёт длину сжатой
последовательности; без него длина строится из предсказанных посимвольных
длительностей. `--temperature 0` даёт детерминированную диагностику,
`0.7-1.1` добавляет вариативность почерка.

## Настройки под GTX 1660

В `configs/gtx1660.toml` уже стоят осторожные параметры:

- маленький `batch_size = 2`;
- gradient accumulation;
- AMP;
- `persistent_workers = true`, чтобы DataLoader workers не пересоздавались
  между эпохами;
- `eval_every = 5`, чтобы validation и запись чекпойнтов шли раз в 5 эпох, а
  не после каждой эпохи;
- ограничение очень длинных примеров через `data.max_points`;
- небольшой text encoder и Conformer flow в latent-пространстве.

Если не хватает VRAM:

1. поставить `data.batch_size = 1`;
2. увеличить `grad_accum_steps`;
3. уменьшить `data.max_points`;
4. уменьшить `generator.hidden_dim` и `generator.layers`.

Если обучение идёт стабильно и памяти хватает, можно постепенно увеличивать
`data.max_points`, чтобы вернуть самые длинные строки.

Если между эпохами всё ещё есть длинная пауза:

1. увеличить `eval_every`, например до `10`;
2. убедиться, что `persistent_workers = true` при `num_workers > 0`;
3. уменьшить частоту `checkpoint_every`;
4. уменьшить `num_workers`, если CPU перегружен, или увеличить до `4`, если
   подготовка батчей не успевает за GPU.

## Как понять, что всё работает

Порядок диагностики такой:

1. `autoencoder` должен хорошо восстанавливать реальные строки.
2. `generator` должен сначала научиться уверенно переобучаться на train.
3. `recognizer` должен снижать CTC loss и позже использоваться для CER/rerank.
4. Сгенерированный `.json` должен рендериться без резких скачков и странных
   подъёмов пера.

Минимальный чек-лист между этапами:

```bash
# 1. Датасет и preprocessing
.venv/bin/python main_training.py --stage audit

# 2. После autoencoder
.venv/bin/python main_evaluate.py --stage autoencoder --split val --count 8

# 3. После generator: сначала train, потом val
.venv/bin/python main_evaluate.py --stage generator --split train --count 8 --generator-selection train_best
.venv/bin/python main_evaluate.py --stage generator --split val --count 8

# 4. После recognizer
.venv/bin/python main_evaluate.py --stage recognizer
```

## Если генерация нечитаемая

Сначала определить, где именно проблема.

1. Если `autoencoder` плохо восстанавливает реальные строки, нужно улучшать
   автоэнкодер: больше latent-размер, меньше downsample, дольше обучение или
   меньше `kl_weight`.

2. Если `autoencoder` хороший, но `generator` плохо пишет даже `--split train`
   с реальной latent-длиной, нужно смотреть метрики `alignment`, `duration` и
   `velocity`: сначала должен стабилизироваться MAS, затем flow.

3. Если `generator` пишет train, но плохо пишет val, проблема уже в
   обобщении: нужно больше данных, лучше покрытие буквосочетаний и аккуратнее
   регуляризация.

В текущей версии генератор обучается как `AlignedLatentFlow`: MAS +
посимвольные длительности + rectified flow в latent-пространстве. Если
генератор был обучен предыдущей архитектурой или версией до
`generator_training_version = 6`, его нужно переобучить:

```bash
.venv/bin/python main_training.py --stage generator
```

Автоэнкодер при этом можно не переобучать, если его реконструкции выглядят
достаточно хорошо.

`main_generate.py` специально не будет молча использовать старый чекпойнт:
если увидишь сообщение `Generator checkpoint is outdated or unsupported`,
просто переобучи generator командой выше. Старый flow-чекпойнт можно открыть
только явно:

```bash
.venv/bin/python main_generate.py "Текст" --allow-legacy-flow
```

Это нужно только для сравнения/отладки, не для нормальной генерации.

Для проверки датасета:

```bash
.venv/bin/python main_training.py --stage audit
```

Для тестов:

```bash
.venv/bin/python -m unittest discover -s tests
```

## Низкоуровневый CLI

Корневые скрипты используют пакетный CLI внутри `src/handwriting_ai/`. Если
понадобится точечный запуск, команды всё ещё доступны:

```bash
PYTHONPATH=src .venv/bin/python -m handwriting_ai audit-data --config configs/gtx1660.toml
PYTHONPATH=src .venv/bin/python -m handwriting_ai render --json-path outputs/generated/sample.json --out outputs/generated/sample.png
```

## Передача эпох по локальной сети

`main_sync.py` - отдельная утилита только для передачи файлов эпох/чекпойнтов.
Она не синхронизирует весь проект, датасет или исходный код. По умолчанию она
работает только с папкой `runs/`, которая лежит рядом с `main_sync.py`.

На ПК-приёмнике, куда нужно складывать эпохи:

```bash
python main_sync.py serve --token my_secret
```

Сервер примет файлы в папку:

```text
runs/
```

На ПК с видеокартой, где идёт обучение:

```bash
python main_sync.py watch --remote http://192.168.1.20:8765 --token my_secret
```

Замени `192.168.1.20` на IP ПК-приёмника. Скрипт будет раз в минуту смотреть
в локальную папку `runs/` и отправлять новые или изменённые файлы:

- `best.pt`;
- `train_best.pt`;
- `last.pt`;
- `epoch_*.pt`;
- `config.json`.

Если нужны только `best.pt`, `train_best.pt`, `last.pt` и `config.json`, без промежуточных эпох:

```bash
python main_sync.py watch --remote http://192.168.1.20:8765 --token my_secret --best-last-only
```

Разовая отправка вместо постоянного наблюдения:

```bash
python main_sync.py push --remote http://192.168.1.20:8765 --token my_secret
```

Если по какой-то причине конфиги передавать не нужно:

```bash
python main_sync.py watch --remote http://192.168.1.20:8765 --token my_secret --no-configs
```

Если папка эпох нестандартная, можно явно указать её на обоих ПК:

```bash
python main_sync.py serve --epochs-dir /Users/frimo/Documents/PycharmProjects/Handwriting/runs --token my_secret
python main_sync.py watch --epochs-dir /Users/frimo/Documents/PycharmProjects/Handwriting/runs --remote http://192.168.1.20:8765 --token my_secret
```

Файлы, которые изменялись меньше `--min-age` секунд назад, не отправляются,
чтобы не копировать недописанный чекпойнт. По умолчанию `--min-age 5`.

В логе передачи видно, что именно произошло:

```text
Scan 14:25:03: /.../runs | files=4 | mode=best/last only | configs=yes
FILE gtx1660/generator/config.json | 1.8 KB
UP   gtx1660/generator/config.json | 1.8 KB | 0.01s | 0.17 MB/s
FILE gtx1660/generator/best.pt | 63.0 MB
UP   gtx1660/generator/best.pt | 63.0 MB | 4.21s | 14.96 MB/s
FILE gtx1660/generator/train_best.pt | 63.0 MB
UP   gtx1660/generator/train_best.pt | 63.0 MB | 4.18s | 15.07 MB/s
FILE gtx1660/generator/last.pt | 63.0 MB
SKIP gtx1660/generator/last.pt | 63.0 MB | already on receiver
Done. uploaded=3, skipped=1, found=4, sent=126.0 MB, elapsed=8.45s, avg=14.91 MB/s
```

`UP` означает, что файл передан. `SKIP` означает, что такой же файл уже есть
на принимающем ПК. Скорость считается по каждому переданному файлу и по всему
проходу.

На принимающем ПК во время загрузки будет строка вида:

```text
RECV gtx1660/generator/best.pt | 63.0 MB | 4.18s | 15.07 MB/s
```
