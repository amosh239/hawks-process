# Текущая GBDT-модель на `dayuses`

Референсный прогон:

- `new_results/orbitals-dayuses-hawkes-20260401-132748-118967`

Текущее качество global GBDT:

$$
\text{mean ll\_gain} = 4.685912
$$

`new_results/orbitals-dayuses-hawkes-20260401-132748-118967/metrics.json:121`

## Что именно сейчас реализовано

- Один объект обучения = один пользователь в один день.
- Target = число заказов `to_ord[t]` в текущий день.
- В признаки попадает только история до дня `t-1`: `src/daylevel_gbdt.py:27`
- Обучение идет через `HistGradientBoostingRegressor(loss="poisson")`: `src/daylevel_gbdt.py:143`

## Формально

Модель строит прогноз

$$
\hat{\lambda}_{u,t} = f\!\left(\mathcal{H}_{u,t-1}\right)
$$

где:

- $u$ - пользователь,
- $t$ - день,
- $\mathcal{H}_{u,t-1}$ - вся доступная история пользователя до предыдущего дня.

Итоговая интенсивность интерпретируется как ожидаемое число заказов в день $t$.

## Какие признаки используются

### 1. Календарные

- `dow`
- `days_seen`
- `week_idx`

`src/daylevel_gbdt.py:30`

### 2. История по каждому исходному сигналу

Для всех `DAYUSES_FEATURES`:

- `*_yesterday`
- `*_sum3`
- `*_sum7`
- `*_mean7`
- `*_active_days7`
- `*_exp3`
- `*_last_active`
- `*_days_since_last_active`

`src/daylevel_gbdt.py:35`

Исходные сигналы берутся из:

- `search`
- `cat`
- `searches`
- `has_search_to_cart`
- `has_search_to_ord`
- `has_cat_to_cart`
- `has_cat_to_ord`
- `search_to_cart`
- `search_to_ord`
- `cat_to_cart`
- `cat_to_ord`
- `to_cart`
- `to_ord`
- `gmv`

`src/datasets/orbitals_dayuses.py:7`

### 3. Общие агрегаты по истории заказов и активности

- `orders_hist_mean`
- `orders_hist_purchased_days`
- `carts_hist_mean`
- `any_activity_yesterday`
- `any_activity_sum7`
- `any_activity_mean7`
- `any_activity_active_days7`
- `days_since_last_any_activity`

`src/daylevel_gbdt.py:59`

### 4. EMA-признаки

- `to_ord_ewm7`
- `to_ord_ewm28`
- `to_ord_ewm_gap_7_28`
- `to_cart_ewm7`
- `to_cart_ewm28`
- `to_cart_ewm_gap_7_28`
- `any_activity_ewm7`
- `any_activity_ewm28`
- `any_activity_ewm_gap_7_28`

`src/daylevel_gbdt.py:74`

### 5. Recency-признаки

- `days_since_last_order`
- `days_since_last_cart`
- `days_since_last_search`
- `days_since_last_cat`

`src/daylevel_gbdt.py:83`

### 6. Простые funnel-ratio признаки

- `ord_per_cart_7`
- `search_to_cart_rate_7`
- `search_to_ord_rate_7`
- `cat_to_cart_rate_7`
- `cat_to_ord_rate_7`

`src/daylevel_gbdt.py:95`

## Как считается test

- Для каждого пользователя train/test делится по времени.
- На test каждый следующий день предсказывается отдельно.
- При прогнозе дня $t$ используются уже наблюденные дни до $t-1$, включая более ранние дни test.

Это реализовано через:

- построение test features: `src/daylevel_gbdt.py:123`
- последовательное формирование признаков по истории: `src/daylevel_gbdt.py:132`

Это online-постановка, а не leakage.

## Что еще есть поверх raw GBDT

Кроме `ll_gbdt`, в раннере сейчас еще считаются:

- `ll_gbdt_blend` - смесь baseline и GBDT с весом от `orders_train`: `scripts/run_dayuses_hawkes.py:271`
- `ll_gbdt_selector` - простой selector между raw GBDT и blend по порогу `orders_train`: `scripts/run_dayuses_hawkes.py:318`

## Самые важные признаки в текущем full-run

Топ importance:

- `orders_hist_mean`: `new_results/orbitals-dayuses-hawkes-20260329-231557/feature_importance_gbdt_global.csv:2`
- `gmv_exp3`: `new_results/orbitals-dayuses-hawkes-20260329-231557/feature_importance_gbdt_global.csv:3`
- `to_ord_ewm28`: `new_results/orbitals-dayuses-hawkes-20260329-231557/feature_importance_gbdt_global.csv:4`
- `days_seen`: `new_results/orbitals-dayuses-hawkes-20260329-231557/feature_importance_gbdt_global.csv:5`
- `any_activity_ewm28`: `new_results/orbitals-dayuses-hawkes-20260329-231557/feature_importance_gbdt_global.csv:6`

## Короткий вывод

Текущая GBDT-модель - это глобальная day-level Poisson regression на богатом наборе history-features:

- календарь,
- недавняя активность,
- EMA-тренды,
- recency,
- простые funnel ratios,
- денежный сигнал через `gmv`.

На данный момент это сильнейшая модель на `dayuses` из всех проверенных.
