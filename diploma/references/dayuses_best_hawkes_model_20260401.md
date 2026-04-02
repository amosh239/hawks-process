# Лучшая Hawkes-модель на `dayuses`

Референсный прогон: `new_results/orbitals-dayuses-hawkes-20260401-132748-118967`

## Разбиение на train / test

- Для каждого пользователя временной ряд режется по времени: первые `70%` дней идут в `train`, последние `30%` - в `test`: `scripts/run_dayuses_hawkes.py:105`
- Все параметры модели обучаются только на `train`: `scripts/run_dayuses_hawkes.py:247`

## Бейзлайн

Для пользователя $u$ в день $t$ базовая интенсивность задается так:

$$
\lambda_{\text{base}}(u,t) = \mu_u \, s_{\operatorname{dow}(t)}
$$

где:

- $s_{\operatorname{dow}(t)}$ - общий профиль по дню недели, обученный на всех train-днях: `src/daylevel_hawkes.py:11`
- $\mu_u$ - пользовательский масштаб, оцененный по MLE на train-окне этого пользователя: `src/daylevel_hawkes.py:22`

## Победившая Hawkes-модель

Лучшая Hawkes-style модель - это pooled additive multi-kernel Hawkes:

$$
\lambda(u,t) = \mu_u \, s_{\operatorname{dow}(t)} + \sum_{j=1}^{J} \sum_{m=1}^{M} \alpha_{j,m} \, z_{u,j,m,t}
$$

где:

- $j$ пробегает признаки:
  - `searches`
  - `search_to_cart`
  - `search_to_ord`
  - `cat_to_cart`
  - `cat_to_ord`
  - `to_cart`
  - `to_ord`
- $m$ пробегает набор half-life:
  - $1$, $3$, $7$, $21$ дней: `new_results/orbitals-dayuses-hawkes-20260401-132748-118967/metrics.json:29`

## Ядро и состояния

Состояние Hawkes-ядра для признака $j$ и масштаба $m$:

$$
z_{u,j,m,t} = \sum_{\tau < t} x_{u,j,\tau} \exp\left(-\beta_m (t-\tau)\right)
$$

где

$$
\beta_m = \frac{\ln 2}{h_m}
$$

а $h_m$ - соответствующий half-life.

Эквивалентная рекуррентная форма:

$$
z_{u,j,m,t} = e^{-\beta_m} z_{u,j,m,t-1} + x_{u,j,t-1}
$$

- Рекурсия реализована в `src/daylevel_hawkes.py:70`
- Предсказание в день $t$ использует только историю до $t-1$: `src/daylevel_hawkes.py:150`

## Как обучается модель

- Глобальные коэффициенты $\alpha_{j,m}$ обучаются один раз на pooled train-данных всех пользователей: `src/daylevel_hawkes.py:97`
- В раннер в fit передаются только train-блоки: `scripts/run_dayuses_hawkes.py:291`
- Оптимизируется пуассоновский log-likelihood с $L_2$-регуляризацией по $\alpha$: `src/daylevel_hawkes.py:119`

## Какой код используется

Основные классы и функции:

- `src/daylevel_hawkes.py:54` - старый `DayLevelHawkes` (per-user baseline Hawkes для сравнения)
- `src/daylevel_hawkes.py:55` - `PooledBasisHawkesResult` (результат новой pooled basis Hawkes-модели)
- `src/daylevel_hawkes.py:97` - `fit_pooled_basis_hawkes(...)` (обучение глобальных basis-коэффициентов)
- `src/daylevel_hawkes.py:150` - `predict_pooled_basis_lambda(...)` (предсказание интенсивности новой модели)
- `src/daylevel_hawkes.py:11` - `fit_dayofweek_profile(...)` (общий seasonal профиль по дням недели)
- `src/daylevel_hawkes.py:22` - `fit_scaled_baseline(...)` (пользовательский масштаб $\mu_u$)
- `src/daylevel_hawkes.py:29` - `poisson_loglik(...)` (целевая метрика log-likelihood)

Основной раннер:

- `scripts/run_dayuses_hawkes.py:158` - `fit_user_record(...)` для старого per-user Hawkes
- `scripts/run_dayuses_hawkes.py:287` - сбор параметров pooled basis Hawkes
- `scripts/run_dayuses_hawkes.py:301` - запуск `fit_pooled_basis_hawkes(...)`
- `scripts/run_dayuses_hawkes.py:319` - расчет test log-likelihood для новой Hawkes-модели

Подготовка признаков для Hawkes-каналов:

- `scripts/run_dayuses_hawkes.py:118` - `build_feature_matrix(...)`
- В текущем лучшем прогоне используются каналы:
  - `searches`
  - `search_to_cart`
  - `search_to_ord`
  - `cat_to_cart`
  - `cat_to_ord`
  - `to_cart`
  - `to_ord`
  - это зафиксировано в `new_results/orbitals-dayuses-hawkes-20260401-132748-118967/metrics.json:20`

## Почему здесь нет leakage

- В fit pooled Hawkes не попадают значения из `test`; используются только train-блоки: `scripts/run_dayuses_hawkes.py:292`
- Состояние на дне $t$ зависит только от $x_{t-1}, x_{t-2}, \dots$, но не от текущего дня и не от будущего: `src/daylevel_hawkes.py:77`
- Test log-likelihood считается только на отложенном хвосте: `scripts/run_dayuses_hawkes.py:319`

Важный нюанс:

- это online-оценка;
- для более поздних дней в `test` модель использует уже наблюденные предыдущие дни из `test`.

Это не leakage, а нормальный режим последовательного обновления интенсивности.

## Итог по качеству

- Старый per-user Hawkes:

$$
\text{mean ll\_gain} = 0.941478
$$

`new_results/orbitals-dayuses-hawkes-20260401-132748-118967/metrics.json:85`

- Новый pooled basis Hawkes:

$$
\text{mean ll\_gain} = 3.034611
$$

`new_results/orbitals-dayuses-hawkes-20260401-132748-118967/metrics.json:108`

- Global `GBDT` все еще сильнее:

$$
\text{mean ll\_gain} = 4.685912
$$

`new_results/orbitals-dayuses-hawkes-20260401-132748-118967/metrics.json:122`

## Короткий вывод

На `dayuses` Hawkes удалось заметно улучшить не через per-user fit, а через:

- pooled обучение по всем пользователям,
- несколько временных масштабов памяти,
- сохранение отдельного сезонного бейзлайна $\mu_u s_{\operatorname{dow}(t)}$,
- добавление Hawkes-компоненты как краткосрочного триггерного эффекта поверх бейзлайна.
