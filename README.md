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

2. `LatentRegressorTransformer`

   Генерирует latent-последовательность по тексту. Это supervised Transformer:
   он напрямую учится предсказывать latent-векторы автоэнкодера по строке
   текста и позиции внутри будущей траектории. Для маленького датасета это
   надёжнее, чем flow/diffusion: модель должна сначала научиться писать хотя
   бы train-примеры.

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
.venv/bin/python main_generate.py --profile cpu_smoke "Тестовая строка." --steps 2 --latent-length 8
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

Открой картинки в `outputs/eval/autoencoder/`. Реконструкция должна быть
похожа на оригинал почти до уровня отдельных букв. Если она превращает буквы в
палочки или ломает соединения, генератор дальше не спасёт ситуацию.

Когда появится `runs/gtx1660/autoencoder/best.pt`, можно обучать генератор:

```bash
.venv/bin/python main_training.py --stage generator
```

После генератора проверить хотя бы train и val:

```bash
.venv/bin/python main_evaluate.py --stage generator --split train --count 8
.venv/bin/python main_evaluate.py --stage generator --split val --count 8
```

По умолчанию `main_evaluate.py` для генератора берёт длину latent-последовательности
из реального примера. Это упрощает диагностику: если даже при правильной длине
строка нечитаемая, значит генератор не научился соответствию `текст -> латенты`.
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
.venv/bin/python main_generate.py "Фиксированная длина." --latent-length 128
.venv/bin/python main_generate.py "Именованный файл." --name my_sample
```

Файл `.json` содержит траекторию для плоттера, `.png` нужен только для
быстрого визуального контроля.

Для текущего supervised-генератора `--steps` не влияет на результат; этот
параметр нужен только для старых flow-чекпойнтов. `--temperature` добавляет
небольшой шум в latent-пространстве, поэтому для диагностики лучше ставить
`--temperature 0`.

## Настройки под GTX 1660

В `configs/gtx1660.toml` уже стоят осторожные параметры:

- маленький `batch_size = 2`;
- gradient accumulation;
- AMP;
- ограничение очень длинных примеров через `data.max_points`;
- небольшой Transformer.

Если не хватает VRAM:

1. поставить `data.batch_size = 1`;
2. увеличить `grad_accum_steps`;
3. уменьшить `data.max_points`;
4. уменьшить `generator.hidden_dim` и `generator.layers`.

Если обучение идёт стабильно и памяти хватает, можно постепенно увеличивать
`data.max_points`, чтобы вернуть самые длинные строки.

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
.venv/bin/python main_evaluate.py --stage generator --split train --count 8
.venv/bin/python main_evaluate.py --stage generator --split val --count 8

# 4. После recognizer
.venv/bin/python main_evaluate.py --stage recognizer
```

## Если генерация нечитаемая

Сначала определить, где именно проблема.

1. Если `autoencoder` плохо восстанавливает реальные строки, нужно улучшать
   автоэнкодер: больше latent-размер, меньше downsample, дольше обучение или
   меньше `kl_weight`.

2. Если `autoencoder` хороший, но `generator` плохо пишет даже `--split train`,
   значит генератор не научился попадать в latent-пространство.

3. Если `generator` пишет train, но плохо пишет val, проблема уже в
   обобщении: нужно больше данных, лучше покрытие буквосочетаний и аккуратнее
   регуляризация.

В текущей версии генератор обучается как supervised latent-регрессор с
нормализацией latent-пространства автоэнкодера. Если генератор был обучен
старой flow-версией или старой версией без `latent_normalization`, его нужно
переобучить:

```bash
.venv/bin/python main_training.py --stage generator
```

Автоэнкодер при этом можно не переобучать, если его реконструкции выглядят
достаточно хорошо.

`main_generate.py` специально не будет молча использовать старый flow-чекпойнт:
если увидишь сообщение `Generator checkpoint is legacy or unsupported`, просто
переобучи generator командой выше. Старый чекпойнт можно открыть только явно:

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
