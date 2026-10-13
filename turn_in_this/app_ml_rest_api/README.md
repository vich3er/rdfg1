# Housing Price REST API

ML-сервіс прогнозу ціни житла на датасеті `data/Housing.csv` (545 рядків, 12 ознак).

## Структура
```
api/              FastAPI: main.py (ендпоінти), schemas.py (валідація)
src/              config.py (ознаки/шляхи), train.py (навчання)
models/           model.joblib, metrics.json
data/             Housing.csv
notebooks/        eda_and_training.ipynb
streamlit_app/    app.py (UI)
tests/            pytest
```

## Швидкий старт
```bash
pip install -r requirements.txt
python -m src.train                 # (модель вже в models/, можна перетренувати)
uvicorn api.main:app --reload       # http://localhost:8000/docs
streamlit run streamlit_app/app.py  # http://localhost:8501
pytest
```

## Ендпоінти
| Метод | Шлях | Опис |
|---|---|---|
| GET | `/health` | статус |
| GET | `/model/info` | метрики моделі |
| POST | `/predict` | прогноз для одного об'єкта |
| POST | `/predict/batch` | прогноз для списку `{"items": [...]}` |

Приклад:
```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{
  "area":7420,"bedrooms":4,"bathrooms":2,"stories":3,"mainroad":"yes",
  "guestroom":"no","basement":"no","hotwaterheating":"no","airconditioning":"yes",
  "parking":2,"prefarea":"yes","furnishingstatus":"furnished"}'
```

## Docker
```bash
docker build -t housing-api . && docker run -p 8000:8000 housing-api
```

## Модель
Порівнюються Ridge і GradientBoosting; зберігається краща за R² (метрики в `models/metrics.json`).
