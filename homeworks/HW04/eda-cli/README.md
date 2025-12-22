# S04 – eda_cli: api

Небольшое CLI-приложение для базового анализа CSV-файлов.
Используется в рамках Семинара 03 курса «Инженерия ИИ».

## Требования

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) установлен в систему

## Инициализация проекта

В корне проекта (S03):

```bash
uv sync
```

Эта команда:

- создаст виртуальное окружение `.venv`;
- установит зависимости из `pyproject.toml`;
- установит сам проект `eda-cli` в окружение.

## Запуск CLI

### Краткий обзор

```bash
uv run eda-cli overview data/example.csv
```

Параметры:

- `--sep` – разделитель (по умолчанию `,`);
- `--encoding` – кодировка (по умолчанию `utf-8`).

### Полный EDA-отчёт

```bash
uv run eda-cli report data/example.csv --out-dir reports
```

В результате в каталоге `reports/` появятся:

- `report.md` – основной отчёт в Markdown;
- `summary.csv` – таблица по колонкам;
- `missing.csv` – пропуски по колонкам;
- `correlation.csv` – корреляционная матрица (если есть числовые признаки);
- `top_categories/*.csv` – top-k категорий по строковым признакам;
- `hist_*.png` – гистограммы числовых колонок;
- `missing_matrix.png` – визуализация пропусков;
- `correlation_heatmap.png` – тепловая карта корреляций.

## Тесты

```bash
uv run pytest -q
```

## Добавленные команды

```bash
uv run eda-cli report data/example.csv --out-dir reports --title title --min-missing-share 0.5
```

В результате title - будет заголовком в report.md, min_missing_share задаст порог для отчета по пропускам

## API
Добавил эндпоинт quality-flags-from-csv, который принимает CSV-файл и возвращает полный набор флагов качества
<img width="467" height="242" alt="image" src="https://github.com/user-attachments/assets/b204b576-69ba-4243-b81b-8eb7fe021223" />
Запуск <br> 
```bash 
uv run uvicorn eda_cli.api:app --reload --port 8000
```
