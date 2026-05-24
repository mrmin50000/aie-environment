# Классификация дефектов поверхности стали

Сервис для классификации дефектов поверхности стали по изображению.  
Используется датасет **NEU Surface Defect Database** (NEU-DET) — 6 классов дефектов, 1800 изображений 200×200.

Модели: **ResNet18** / **MobileNetV2** с transfer learning.

---

## 1. Структура проекта

```
project/
├── configs/
│   ├── config.yaml         # Параметры обучения и сервиса
│   └── .env.example        # Шаблон переменных окружения
├── data/
│   ├── train/              # Обучающая выборка (1440 изобр.)
│   └── validation/         # Валидационная выборка (360 изобр.)
├── notebooks/
│   ├── 01_eda.ipynb        # Разведочный анализ данных
│   ├── 02_baselines.ipynb  # Baseline + transfer learning
│   └── 03_experiments.ipynb# Эксперименты и тюнинг
├── src/
│   ├── data/
│   │   ├── dataset.py      # Датасет и аугментации
│   │   └── split.py        # Стратифицированное разбиение
│   ├── models/
│   │   └── model.py        # Создание моделей (ResNet, MobileNetV2)
│   ├── service/
│   │   └── app.py          # FastAPI сервис
│   ├── train.py            # Цикл обучения
│   └── predict.py          # Инференс
├── tests/
│   ├── test_data.py        # Тесты данных
│   ├── test_model.py       # Тесты модели
│   └── test_service.py     # Тесты сервиса
├── artifacts/              # Сохранённые модели и логи
├── requirements.txt        # Зависимости
├── README.md               # Документация
├── report.md               # Отчёт по проекту
└── self-checklist.md       # Чеклист самопроверки
```

---

## 2. Требования и установка

### 2.1. Требования

- Python >= 3.10
- CUDA-совместимый GPU (рекомендуется) или CPU

### 2.2. Установка

```bash
cd project
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> Если torch/torchvision не ставятся, установите их вручную с pytorch.org под вашу платформу:
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
> ```

---

## 3. Как запустить

### 3.1. Обучение модели

```bash
cd project
source .venv/bin/activate
python -m src.train --config configs/config.yaml
```

Результат: `artifacts/best_model.pth`

### 3.2. Запуск сервиса

```bash
cd project
source .venv/bin/activate
python -m src.service.app
```

Сервис поднимается на `http://0.0.0.0:8000`

### 3.3. Эндпоинты

| Метод | Путь | Описание |
|-------|------|----------|
| GET | `/health` | Проверка состояния сервиса |
| POST | `/predict` | Классификация изображения (multipart/form-data) |

### 3.4. Пример запроса

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@data/validation/images/scratches/scratches_241.jpg"
```

Пример ответа:

```json
{
  "class": "scratches",
  "probabilities": {
    "crazing": 0.0012,
    "inclusion": 0.0035,
    "patches": 0.0081,
    "pitted_surface": 0.0023,
    "rolled-in_scale": 0.0049,
    "scratches": 0.9800
  }
}
```

### 3.5. Инференс из командной строки

```bash
python -m src.predict path/to/image.jpg
```

---

## 4. Тесты

```bash
cd project
source .venv/bin/activate
python -m pytest tests -v
```

---

## 5. Данные

- **Датасет:** NEU Surface Defect Database (NEU-DET)
- **Классы:** crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches
- **Размер:** 1800 изображений (200×200, grayscale)
- **Разбивка:** train 1440, validation 360 (сбалансировано по классам)
- **Формат разметки:** Pascal VOC XML (bounding boxes)

---

## 6. Демонстрация на защите

1. Запуск сервиса на локальной машине
2. Демонстрация `/health`
3. 2-3 запроса к `/predict` через Swagger UI и curl
4. Показ ноутбука `02_baselines.ipynb` со сравнением моделей
