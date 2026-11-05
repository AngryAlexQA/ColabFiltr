# Коллаборативная фильтрация на MovieLens 100K

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AngryAlexQA/ColabFiltr/blob/main/collaborative_filtering.ipynb)

Практическое задание «Коллаборативная фильтрация»: рекомендательная система на библиотеке [`surprise`](https://surpriselib.com/) и датасете MovieLens 100K. Сравниваются два классических подхода — **User-based** (сходство пользователей) и **Item-based** (сходство фильмов) — от анализа данных до проверки рекомендаций для конкретного пользователя.

**Автор:** Александра Бужор (Alexandra Bujor) · Telegram: [@Angry_C_F_O](https://t.me/Angry_C_F_O)

**Решение:** [`collaborative_filtering.ipynb`](collaborative_filtering.ipynb) — ноутбук выполнен целиком, все графики, таблицы и выводы сохранены в нём.

## Что сделано

| Этап | Содержание | Раздел ноутбука |
|---|---|---|
| 1. Загрузка и подготовка | `Dataset.load_builtin('ml-100k')`, проверка качества данных, сборка `Dataset` через `Reader` | Этап 1 |
| 2. EDA | Распределение оценок, активность пользователей, популярность фильмов, «длинный хвост», **анализ разреженности** (заполненность матрицы, число общих оценок у пар пользователей и фильмов) | 2.1–2.5 |
| 3. Модели | `KNNBasic` с косинусной мерой: User-based и Item-based; диагностика матриц сходства | Этап 3 |
| 4. Оценка качества | Кросс-валидация на 5 одинаковых фолдах, RMSE и MAE, ручной расчёт метрик, анализ ошибок, сравнение с бейзлайнами и классификация моделей по KPI, чувствительность к `k` | 4.1–4.4 |
| 5. Проверка «в полях» | Рекомендации для случайного пользователя с интерпретацией, проверка надёжности и разнообразия top-10 на 200 пользователях, итоговая таблица, выводы | 5.1–5.4, «Итоговая таблица», «Выводы» |

## Ключевые результаты

Кросс-валидация на 5 фолдах (среднее ± стандартное отклонение):

| Модель | Подход | RMSE | MAE | KPI |
|---|---|---|---|---|
| **KNNBasic · cosine (задание)** | **User-based** | **1.0171 ± 0.0047** | **0.8044 ± 0.0048** | Неэффективна |
| **KNNBasic · cosine (задание)** | **Item-based** | **1.0255 ± 0.0033** | **0.8110 ± 0.0037** | Неэффективна |
| KNNBaseline · pearson_baseline | Item-based | 0.9155 ± 0.0038 | 0.7182 ± 0.0033 | Эффективна |
| KNNBaseline · pearson_baseline | User-based | 0.9215 ± 0.0036 | 0.7210 ± 0.0031 | Эффективна |
| SVD (параметры по умолчанию) | Model-based | 0.9350 ± 0.0036 | 0.7372 ± 0.0026 | Эффективна |
| BaselineOnly (μ + b_u + b_i) | Бейзлайн | 0.9436 ± 0.0037 | 0.7481 ± 0.0036 | ориентир KPI |
| Глобальное среднее | Бейзлайн | 1.1257 ± 0.0054 | 0.9447 ± 0.0050 | ориентир |

KPI: модель **эффективна**, если её RMSE и MAE ниже, чем у `BaselineOnly`, на каждом из 5 фолдов. Полная таблица из 14 моделей — в разделе 4.3 ноутбука.

**Выводы:**
1. Матрица «пользователь × фильм» разрежена на **93,7 %**. У пар пользователей медиана — 10 общих фильмов, у пар фильмов — всего 1 общий зритель, поэтому сходство фильмов вычисляется по очень малой поддержке.
2. В конфигурации задания **точнее User-based**: он выигрывает у Item-based на всех 5 фолдах по RMSE и MAE и при любом `k` от 5 до 160.
3. Обе модели из задания по KPI **неэффективны** — уступают простой модели смещений `BaselineOnly`: косинус на ненормированных оценках почти не различает соседей (среднее сходство ≈ 0,95).
4. С нормализацией вывод меняется: лучшей становится **Item-based `KNNBaseline` + `pearson_baseline`** (RMSE 0,9155) — точнее и User-based, и `SVD` по умолчанию.
5. Проверка «в полях»: User-based из задания при лучшем RMSE рекомендует всем одни и те же фильмы с единственной «пятёркой» (99,5 % редких фильмов в top-10 у 200 пользователей). RMSE и MAE нужно дополнять проверкой самих списков рекомендаций.

## Как запустить

**Google Colab.** Нажмите кнопку «Open in Colab» вверху и выберите «Среда выполнения → Выполнить всё». Библиотека `scikit-surprise` установится автоматически в первой ячейке.

**Локально (Windows, PowerShell):**
```powershell
git clone https://github.com/AngryAlexQA/ColabFiltr.git
cd ColabFiltr
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
jupyter notebook collaborative_filtering.ipynb
```

**Локально (macOS / Linux):**
```bash
git clone https://github.com/AngryAlexQA/ColabFiltr.git
cd ColabFiltr
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
jupyter notebook collaborative_filtering.ipynb
```

Полный прогон занимает около 6 минут; больше всего времени уходит на кросс-валидацию 14 моделей и перебор `k`.

> Готовые сборки `scikit-surprise` 1.1.5 есть для Windows, Linux и Mac на Apple Silicon. На Mac с процессором Intel pip собирает библиотеку из исходников — нужны Xcode Command Line Tools (`xcode-select --install`); альтернатива — `conda install -c conda-forge scikit-surprise` или запуск в Colab.

## Данные
Используется встроенный датасет `surprise` — [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) (GroupLens Research, University of Minnesota): 100 000 оценок от 943 пользователей для 1 682 фильмов. При первом запуске `Dataset.load_builtin('ml-100k', prompt=False)` скачивает архив в `~/.surprise_data`. По условиям лицензии GroupLens данные в репозиторий не включены.

> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems 5, 4, Article 19. https://doi.org/10.1145/2827872

## Структура репозитория
```
ColabFiltr/
├── collaborative_filtering.ipynb   # решение: код, графики, таблицы, выводы
├── requirements.txt                # зависимости
├── README.md
└── .gitignore
```

Проверено на Python 3.13, scikit-surprise 1.1.5, pandas 3.0, NumPy 2.5; результаты воспроизводимы (`SEED = 42`, фиксированные фолды).
