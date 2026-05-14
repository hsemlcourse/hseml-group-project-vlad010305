[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)

# ML Project — Predictive Maintenance: предсказание отказа промышленного оборудования

**Студент:** Шныренков Владислав Андреевич

**Группа:** БИВ238

---

# Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Запуск проекта](#запуск-проекта)
4. [Docker и воспроизводимость](#docker-и-воспроизводимость)
5. [Linting](#linting)
6. [Данные](#данные)
7. [Обработка данных и feature engineering](#обработка-данных-и-feature-engineering)
8. [Моделирование](#моделирование)
9. [Результаты](#результаты)
10. [Выводы](#выводы)
11. [Отчёт](#отчёт)

---

# Описание задачи

Проект посвящён задаче предиктивного обслуживания промышленного оборудования.

Цель проекта — построить модель бинарной классификации, которая по операционным характеристикам оборудования предсказывает вероятность отказа машины.

Задача предиктивного обслуживания является важной для промышленности, поскольку позволяет:

- заранее обнаруживать потенциальные отказы;
- уменьшать простой оборудования;
- снижать затраты на ремонт;
- повышать надёжность производственных процессов.

## Постановка задачи

- **Тип задачи:** бинарная классификация
- **Целевая переменная:** `Machine failure`
- **Положительный класс:** `1` — отказ оборудования
- **Отрицательный класс:** `0` — отказа оборудования нет

## Датасет

**AI4I 2020 Predictive Maintenance Dataset**

Источник:

https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification

## Метрики качества

Основная метрика:

- **PR-AUC**

Дополнительные метрики:

- ROC-AUC
- F1-score
- Precision
- Recall
- Confusion Matrix

PR-AUC выбрана как основная метрика, поскольку задача является несбалансированной: доля отказов оборудования составляет около 3.4%.

---

# Структура репозитория

```text
.
├── data
│   ├── processed               # Промежуточные данные
│   └── raw                     # Исходные данные
├── models                      # Сохранённые модели
├── notebooks
│   ├── 01_eda.ipynb            # Анализ данных и визуализации
│   ├── 02_baseline.ipynb       # Baseline-модели
│   └── 03_experiments.ipynb    # Эксперименты и hyperparameter tuning
├── presentation                # Презентация
├── report
│   ├── images                  # Изображения для отчёта
│   └── report.md               # Финальный отчёт
├── src
│   ├── preprocessing.py        # Предобработка данных
│   ├── features.py             # Feature engineering
│   └── modeling.py             # Метрики и оценка моделей
├── tests
│   └── test.py                 # Тесты
├── .dockerignore
├── .pre-commit-config.yaml
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

# Запуск проекта

## 1. Клонирование репозитория

```bash
git clone <repo-url>
cd <repo-name>
```

## 2. Переключение на ветку проекта

```bash
git checkout cp2
```

## 3. Создание виртуального окружения

### Windows

```bash
py -m venv .venv
```

### Linux/macOS

```bash
python -m venv .venv
```

## 4. Активация виртуального окружения

### Windows PowerShell

```bash
.venv\Scripts\activate
```

### Linux/macOS

```bash
source .venv/bin/activate
```

## 5. Установка зависимостей

```bash
pip install -r requirements.txt
```

## 6. Запуск ноутбуков

Проект выполняется через Jupyter Notebook в Visual Studio Code.

Основные ноутбуки:

```text
notebooks/01_eda.ipynb
notebooks/02_baseline.ipynb
notebooks/03_experiments.ipynb
```

Рекомендуемый порядок запуска:

1. `01_eda.ipynb` — анализ данных;
2. `02_baseline.ipynb` — baseline-модели;
3. `03_experiments.ipynb` — feature engineering, эксперименты и tuning.

---

# Docker и воспроизводимость

Проект поддерживает запуск через Docker и Docker Compose.

## Сборка Docker-образа

```bash
docker build -t predictive-maintenance .
```

## Запуск через docker-compose

```bash
docker-compose up
```

После запуска Jupyter Notebook будет доступен по адресу:

```text
http://localhost:8888
```

## Остановка контейнеров

```bash
docker-compose down
```

Docker-конфигурация включает:

- `Dockerfile`;
- `docker-compose.yml`;
- автоматическую установку зависимостей;
- запуск Jupyter Notebook внутри контейнера.

---

# Linting

Для проверки качества кода используется `ruff`.

## Проверка линтером

```bash
ruff check src
```

## Автоматическое исправление

```bash
ruff check src --fix
```

Настройки линтера находятся в:

```text
pyproject.toml
```

Также проект поддерживает:

- `Makefile`;
- `pre-commit hooks`.

---

# Данные

Используется датасет:

**AI4I 2020 Predictive Maintenance Dataset**

Файл с исходными данными:

```text
data/raw/ai4i2020.csv
```

## Размер датасета

- 10 000 строк
- 14 признаков

## Основные признаки

| Признак | Описание |
|---|---|
| `Type` | Тип продукта |
| `Air temperature [K]` | Температура воздуха |
| `Process temperature [K]` | Температура процесса |
| `Rotational speed [rpm]` | Скорость вращения |
| `Torque [Nm]` | Крутящий момент |
| `Tool wear [min]` | Износ инструмента |

## Целевая переменная

| Переменная | Описание |
|---|---|
| `Machine failure` | Факт отказа оборудования |

## Исключённые признаки

### Идентификаторы

| Признак | Причина |
|---|---|
| `UDI` | Технический идентификатор |
| `Product ID` | Уникальный ID продукта |

### Leakage-признаки

```text
TWF, HDF, PWF, OSF, RNF
```

Эти признаки напрямую связаны с фактом отказа оборудования и могут приводить к data leakage.

---

# Обработка данных и feature engineering

В проекте были выполнены:

- проверка пропусков;
- проверка дубликатов;
- анализ дисбаланса классов;
- анализ распределений признаков;
- анализ выбросов методом IQR;
- визуализация признаков;
- стратифицированный split 70/15/15;
- удаление leakage-признаков;
- feature engineering.

## Распределение целевой переменной

| Класс | Количество | Доля |
|---|---:|---:|
| `0` — отказа нет | 9661 | 96.61% |
| `1` — отказ | 339 | 3.39% |

## Разделение данных

| Выборка | Класс 0 | Класс 1 |
|---|---:|---:|
| Train | 6763 | 237 |
| Validation | 1449 | 51 |
| Test | 1449 | 51 |

## Новые признаки

| Новый признак | Описание |
|---|---|
| `temperature_diff` | Разница температур |
| `power_proxy` | Приближённая нагрузка |
| `torque_per_rpm` | Отношение момента к скорости |
| `wear_torque_interaction` | Взаимодействие износа и момента |

Наиболее важными признаками стали:

1. `temperature_diff`
2. `power_proxy`
3. `wear_torque_interaction`

---

# Моделирование

## Baseline-модели

- Logistic Regression
- Logistic Regression (`class_weight='balanced'`)

## Экспериментальные модели

- Logistic Regression
- Logistic Regression Balanced
- RandomForestClassifier
- ExtraTreesClassifier
- GradientBoostingClassifier
- HistGradientBoostingClassifier

## Hyperparameter tuning

Для RandomForestClassifier был выполнен систематический подбор гиперпараметров с помощью:

```text
RandomizedSearchCV
```

Использовались:

- `cv=5`;
- `n_iter=15`;
- метрика `average_precision` (PR-AUC).

Подбирались параметры:

- `n_estimators`;
- `max_depth`;
- `min_samples_split`;
- `min_samples_leaf`;
- `max_features`.

---

# Результаты

## Baseline-модели

| Эксперимент | PR-AUC | ROC-AUC | F1 |
|---|---:|---:|---:|
| Logistic Regression | 0.3565 | 0.8427 | 0.2623 |
| Logistic Regression Balanced | 0.3083 | 0.8402 | 0.1918 |

## Лучшие модели после feature engineering

| Модель | PR-AUC | ROC-AUC | F1 |
|---|---:|---:|---:|
| Gradient Boosting FE | 0.8661 | 0.9532 | 0.8132 |
| Random Forest FE | 0.8422 | 0.9569 | 0.8298 |
| HistGradientBoosting FE | 0.8351 | 0.9451 | 0.8298 |

## Финальная tuned-модель

| Модель | Dataset | PR-AUC | ROC-AUC | F1 | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|
| RandomForest Tuned | test | 0.9350 | 0.9791 | 0.9184 | 0.9574 | 0.8824 |

Модель обнаружила:

- 45 из 51 отказов;
- только 2 ложных срабатывания.

---

# Выводы

В рамках проекта были выполнены:

- анализ данных;
- feature engineering;
- обработка дисбаланса классов;
- построение baseline-моделей;
- сравнение нескольких ML-моделей;
- hyperparameter tuning;
- анализ важности признаков;
- финальная оценка на test-выборке.

Основной результат проекта:

```text
Лучшая модель: RandomForest Tuned
PR-AUC: 0.9350
ROC-AUC: 0.9791
F1-score: 0.9184
```

Feature engineering и hyperparameter tuning значительно улучшили качество модели по сравнению с baseline.

---

# Отчёт

Финальный отчёт находится в:

```text
report/report.md
```

Основные ноутбуки проекта:

- `notebooks/01_eda.ipynb`
- `notebooks/02_baseline.ipynb`
- `notebooks/03_experiments.ipynb`