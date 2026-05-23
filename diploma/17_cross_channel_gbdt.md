# 17. Бустинг с полным feature-engineering: верхняя планка для cross-channel

## 17.1. Зачем

Глава 12 показала, что Scaled-baseline Hawkes улучшает Personalized GP на всех 3 каналах воронки (`searches`, `to_cart`, `to_ord`) на `2.92..6.83%` от baseline NLL. Это интерпретируемый выигрыш на узком наборе input'ов (3 Hawkes-states), но он же ставит вопрос: насколько Hawkes-структура **близка к верхней планке** того, что в принципе можно выжать из данных тем же tree-based бустингом с **полным feature-engineering**'ом?

В этой главе обучаем `HistGradientBoostingRegressor` на полном инженерном наборе (`141` фича — те же, что и experimental GBDT из главы 9), отдельно на каждом из 3 каналов как target'е, и смотрим, насколько остаётся «не закрытый» Hawkes'ом потолок.

## 17.2. Протокол

- **Train/test**: те же окна, что в главе 12 (`2025-01-15..2025-08-09` train, `2025-08-10..2025-09-30` test).
- **Фичи**: полный набор из `src.diploma_experimental.gbdt.SOURCE_FEATURES` — 14 source-каналов, на которых строятся `141` инженерный признак:
  - календарь (`dow`, `days_seen`, `week_idx`),
  - history по каждому source: `*_yesterday`, `*_sum3`, `*_sum7`, `*_mean7`, `*_active_days7`, `*_exp3`, `*_last_active`, `*_days_since_last_active`,
  - EMA: `*_ewm7`, `*_ewm28`, `*_ewm_gap_7_28` (для `to_ord`, `to_cart`, `any_activity`),
  - агрегаты по заказам и активности, recency, funnel-ratio'и.
- **Модель**: `HistGradientBoostingRegressor(loss="poisson", max_iter=400, learning_rate=0.05, max_depth=6, min_samples_leaf=200, l2_regularization=1.0, random_state=42, early_stopping=True)`. На каждый канал — отдельный бустинг с теми же гиперпараметрами.
- **Размер train**: `~1.99M` строк, `~10K` пользователей. Размер test: `~513K` строк. Стройка фичей — `427` с на single thread, фит per channel — `19..45` с.

Скрипт: [`run_cross_channel_gbdt_ch25.py`](../scripts/compute/run_cross_channel_gbdt_ch25.py).

## 17.3. Результаты

![Per-channel: Pers GP vs Scaled Hawkes vs GBDT (141 фича)](reports/15_cross_channel_hawkes/main_3ch/baseline_vs_hawkes_nll.png)

| target | Pers. GP | Scaled Hawkes | GBDT (141 фича) | Hawkes − GP | GBDT − Hawkes | GBDT − GP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `searches` | `2.2519` | `2.0980` | `2.0516` | `−0.1539` (−6.83%) | `−0.0464` (−2.21%) | `−0.2003` (−8.89%) |
| `to_cart`  | `1.0321` | `0.9860` | `0.9569` | `−0.0461` (−4.47%) | `−0.0291` (−2.95%) | `−0.0752` (−7.29%) |
| `to_ord`   | `0.4096` | `0.3976` | `0.3877` | `−0.0119` (−2.92%) | `−0.0099` (−2.49%) | `−0.0219` (−5.35%) |

Ладдер «Pers GP → Scaled Hawkes → GBDT (141 фича)» **монотонно убывает по test NLL на всех 3 каналах**.

- Hawkes даёт основной шаг (`2.92..6.83%` от baseline);
- GBDT с полным feature-engineering'ом даёт **дополнительные `2..3%`** сверху;
- Суммарный gap «GP → GBDT» — `5.35..8.89%` от baseline.

На `to_ord` число GBDT (`0.3877`) близко совпадает с experimental GBDT из главы 9 (`0.3881`) — это согласованность с прежним результатом, теперь воспроизведённым в рамках per-channel протокола 12-й главы.

## 17.4. Что показано

- **Hawkes не достигает планки сложной tree-ensemble модели** ни на одном из 3 каналов. Зазор GBDT − Hawkes:
  - на `searches` — `−0.046` нат/n (`−2.21%` от baseline);
  - на `to_cart` — `−0.029` нат/n (`−2.95%`);
  - на `to_ord` — `−0.010` нат/n (`−2.49%`).
- Зазор по абсолютной величине самый большой на плотных каналах (`searches` ≈ 1.2/строка), а в процентах — на `to_cart`. Это согласуется с тем, что GBDT эффективнее ловит нелинейные взаимодействия в плотном сигнале, тогда как линейная additive Hawkes-форма «уплощает» эти взаимодействия.
- При этом **Hawkes остаётся ценен**: он закрывает `2.9..6.8%` зазора от baseline, что составляет **большую часть** общего gap'а GP → GBDT (на `searches` — 77%, на `to_cart` — 61%, на `to_ord` — 54%). Hawkes — экономная и интерпретируемая модель, ловящая лучшую часть сигнала.
- Это согласуется с trade-off из главы 9: GBDT — лучший в абсолюте скор, Hawkes — структурное понимание `(c, μ_u, α)`. Cross-channel постановка не меняет эту картину — она лишь подтверждает её отдельно на каждом из 3 каналов.

Артефакты: [`reports/25_cross_channel_gbdt/`](reports/25_cross_channel_gbdt/), общий график лежит в [`reports/15_cross_channel_hawkes/main_3ch/baseline_vs_hawkes_nll.png`](reports/15_cross_channel_hawkes/main_3ch/baseline_vs_hawkes_nll.png).
