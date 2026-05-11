# 12. Cross-channel Hawkes: структура и bootstrap-стабильность

## 12.1. Зачем

Перенести Hawkes-моделирование из главы 6 (один target `to_ord`, pooled `α`) в **cross-channel** постановку: на каждый из 3 каналов воронки (`searches`, `to_cart`, `to_ord`) подгоняется собственная Hawkes-модель, у которой источниками возбуждения служат **все 3 канала** (включая сам себя). Получается интерпретируемая матрица `α[target ← source]`. Глава отвечает на два вопроса:

1. Какие веса видны на главном `207d`-train у базовой **Scaled-baseline** модели (`λ = c · μ_u · b_t + α^⊤ z`, EB-prior на per-user `μ_u`)?
2. Насколько эти веса **устойчивы** к выбору train-окна — bootstrap по 10 случайным `100d`-окнам, и как картина меняется при переходе к **Joint Hawkes** (`λ = λ_u · b_t + α^⊤ z`) с разными уровнями регуляризации `λ_{ℓ_2}` на per-user multiplier.

## 12.2. Протокол

- **Channels**: `searches` (mean ≈ 1.20/row), `to_cart` (≈ 0.37), `to_ord` (≈ 0.10).
- **Half-life**: `1` день, `n_alpha = 3` source-каналов на target.
- **Main train/test** (раздел 12.3): `2025-01-15..2025-08-09` (207d, ~1.99M строк) / `2025-08-10..2025-09-30` (52d, ~513K).
- **Bootstrap** (разделы 12.4-12.5): 10 случайных `100d` окон из общего диапазона, `66d` train + `34d` test, фиксированный seed.
- **Hyperparameters**: `α_l2 = 1e-4`, `max_iter = 500`. У Joint Hawkes тестируются два уровня регуляризации: `λ_{ℓ_2} = 1.0` (default) и `λ_{ℓ_2} = 0` (без shrinkage'а на `λ_u`).

Скрипты: [`run_cross_channel_hawkes_ch15.py`](../scripts/compute/run_cross_channel_hawkes_ch15.py) (Scaled main train), [`run_cross_channel_hawkes_bootstrap_ch16.py`](../scripts/compute/run_cross_channel_hawkes_bootstrap_ch16.py) (Scaled bootstrap), [`run_cross_channel_joint_hawkes_bootstrap_ch17.py`](../scripts/compute/run_cross_channel_joint_hawkes_bootstrap_ch17.py) (Joint `λ_{ℓ_2}=1` bootstrap), [`run_cross_channel_joint_unregularized_ch18.py`](../scripts/compute/run_cross_channel_joint_unregularized_ch18.py) (Joint `λ_{ℓ_2}=0` bootstrap).

## 12.3. Матрица `α` на главном train (Scaled-baseline)

![scaled main heatmap](reports/15_cross_channel_hawkes/main_3ch/alpha_heatmap.png)

| target ↓ \ source → | `searches` | `to_cart` | `to_ord` |
| --- | ---: | ---: | ---: |
| `searches` | `0.1269` | `0.0240` | `0` |
| `to_cart`  | `0.0103` | `0.0713` | `0` |
| `to_ord`   | `0.0030` | `0.0046` | `0.0194` |

**Верхне-треугольная часть колонки `to_ord` нулевая** (`α[searches←to_ord] = α[to_cart←to_ord] = 0`): после покупки нет краткосрочного Hawkes-сигнала на поиск или корзину — структурный нуль, подтверждаемый во всех последующих фитах. Self-α `α[to_ord←to_ord] = 0.019` ненулевой, но небольшой, и в bootstrap (12.4) опускается практически к нулю. Off-diagonal funnel-связи `searches → cart → order` различимы и положительны.

Артефакты: [`reports/15_cross_channel_hawkes/`](reports/15_cross_channel_hawkes/).

## 12.4. Bootstrap-стабильность (Scaled-baseline, 10 × 100d)

![scaled bootstrap heatmap](reports/16_cross_channel_bootstrap/alpha_heatmap_with_ci.png)

Mean ± std по 10 окнам:

| ячейка | mean `α` | std | CV |
| --- | ---: | ---: | ---: |
| `searches ← searches` | `0.0798` | `0.0058` | `7.2%` |
| `searches ← to_cart`  | `0.0168` | `0.0026` | `15.5%` |
| `to_cart ← searches`  | `0.0060` | `0.0009` | `15.6%` |
| `to_cart ← to_cart`   | `0.0347` | `0.0034` | `9.7%` |
| `to_ord ← searches`   | `0.0011` | `0.0005` | `47.4%` |
| `to_ord ← to_cart`    | `0.0025` | `0.0009` | `36.2%` |
| **`to_ord ← to_ord`** | **`0.0016`** | **`0.0024`** | **`148.3%`** |

**Off-diagonal funnel-связи стабильны** (`CV ≤ 16%` для частых пар, до `47%` для редкого `to_ord ← searches`). **`α[to_ord ← to_ord]` коллапсирует**: median ≈ 0, в части окон значение зажимается к нулю (EB-prior зажимает per-user multiplier к нулю для редко-активных юзеров → у оптимизатора нет signal для self-α редкого target'а). Это и мотивирует переход к Joint Hawkes в 12.5.

Артефакты: [`reports/16_cross_channel_bootstrap/`](reports/16_cross_channel_bootstrap/).

## 12.5. Параметры Joint Hawkes на bootstrap (10 × 100d)

В Joint Hawkes отказ от EB-prior'а: per-user multiplier `λ_u` обучается напрямую, с L2-штрафом `λ_{ℓ_2} · (λ_u - 1)²` к единице. При `λ_{ℓ_2} = 1` штраф активный (зажимает `λ_u` к prior'у), при `λ_{ℓ_2} = 0` штрафа нет вообще — `λ_u` свободно.

### 12.5.1. `λ_{ℓ_2} = 1` (default режим)

![joint l2=1 heatmap](reports/17_cross_channel_joint_bootstrap/alpha_heatmap_with_ci.png)

| ячейка | mean `α` | std | CV |
| --- | ---: | ---: | ---: |
| `searches ← searches` | `0.1020` | `0.0042` | `4.1%` |
| `searches ← to_cart`  | `0.0196` | `0.0017` | `8.7%` |
| `to_cart ← searches`  | `0.0119` | `0.0007` | `5.9%` |
| `to_cart ← to_cart`   | `0.0651` | `0.0033` | `5.1%` |
| `to_ord ← searches`   | `0.0039` | `0.0008` | `21.3%` |
| `to_ord ← to_cart`    | `0.0051` | `0.0010` | `20.0%` |
| **`to_ord ← to_ord`** | **`0.0383`** | **`0.0057`** | **`14.7%`** |

Активный L2-prior `(λ_u - 1)²` **стабилизирует все 9 коэффициентов**, включая `α[to_ord ← to_ord]`: CV `14.7%` против `148%` у Scaled.

### 12.5.2. `λ_{ℓ_2} = 0` (без регуляризации)

![joint l2=0 heatmap](reports/18_joint_unregularized/alpha_heatmap_with_ci.png)

| ячейка | mean `α` | std | CV |
| --- | ---: | ---: | ---: |
| `searches ← searches` | `0.0843` | `0.0068` | `8.0%` |
| `searches ← to_cart`  | `0.0221` | `0.0029` | `13.0%` |
| `to_cart ← searches`  | `0.0122` | `0.0013` | `10.3%` |
| `to_cart ← to_cart`   | `0.0414` | `0.0057` | `13.8%` |
| `to_ord ← searches`   | `0.0035` | `0.0008` | `23.0%` |
| `to_ord ← to_cart`    | `0.0040` | `0.0011` | `26.7%` |
| **`to_ord ← to_ord`** | **`0.0002`** | **`0.0006`** | **`316%`** |

Когда `λ_u` отпущен, **`α[to_ord ← to_ord]` опять коллапсирует** — теперь до `0.0002` (mean) с CV `316%`. Это та же самая картина что у Scaled (12.4): без активной регуляризации на `λ_u`, оптимизатор для редкого target'а зажимает self-α к нулю и компенсирует это другими параметрами. Off-diagonal funnel-связи при этом остаются устойчивыми.

Артефакты: [`reports/17_cross_channel_joint_bootstrap/`](reports/17_cross_channel_joint_bootstrap/), [`reports/18_joint_unregularized/`](reports/18_joint_unregularized/).

<sup>(†) Спектральный радиус mean-α матрицы как branching-ratio: `ρ(α_Scaled) ≈ 0.082`, `ρ(α_Joint, λ_l2=1) ≈ 0.107`, `ρ(α_Joint, λ_l2=0) ≈ 0.094`. Все сильно меньше 1 — Hawkes-процессы далеки от explosion-порога.</sup>

## 12.6. Сравнение: Joint без регуляризации vs Scaled-baseline

![three-way alpha compare](reports/18_joint_unregularized/alpha_compare_three_way.png)

Сравнение mean(α) bootstrap для **Scaled-baseline** (EB-prior зажимает `μ_u` к нулю) и **Joint без рег.** (`λ_u` свободен):

| ячейка | Scaled mean | Joint `λ_l2=0` mean | ratio Joint/Scaled |
| --- | ---: | ---: | ---: |
| `searches ← searches` | `0.0798` | `0.0843` | `×1.06` |
| `searches ← to_cart`  | `0.0168` | `0.0221` | `×1.32` |
| `to_cart ← searches`  | `0.0060` | `0.0122` | `×2.03` |
| `to_cart ← to_cart`   | `0.0347` | `0.0414` | `×1.19` |
| `to_ord ← searches`   | `0.0011` | `0.0035` | `×3.18` |
| `to_ord ← to_cart`    | `0.0025` | `0.0040` | `×1.60` |
| **`to_ord ← to_ord`** | **`0.0016`** | **`0.0002`** | **`×0.13`** |

Структурно картина схожая: off-diagonal funnel-связи различимы и стабильны (ratio в `[1.06, 3.18]`), колонка `to_ord` — нуль, self-`to_ord` коллапсирует. Различие — направление shrinkage'а per-user multiplier'а: Scaled через EB free-shrinks `μ_u` к нулю, Joint без рег. не shrinks вообще, и в обоих случаях у оптимизатора нет structural signal удерживать self-α для редкого target'а.

**Вывод**: Joint Hawkes с **активной** регуляризацией (`λ_{ℓ_2} = 1`) — единственный из 3 рассмотренных режимов, в котором `α[to_ord ← to_ord]` не коллапсирует. Это и мотивирует использовать его как default-режим в следующих главах.

## 12.7. Что унесено в следующие главы

- **Off-diagonal funnel-связи** и **структурный нуль колонки `to_ord`** — фундаментальные структурные результаты, устойчивые ко всем 3 режимам (Scaled, Joint `λ_l2=1`, Joint `λ_l2=0`).
- **`α[to_ord ← to_ord]`** — параметр, удерживаемый только активной регуляризацией. В моделях без неё (Scaled с EB-shrinkage к нулю, Joint `λ_l2=0`) он коллапсирует.
- Joint Hawkes с `λ_{ℓ_2} = 1` берётся как default-режим. Глава 13 разбирает, как сила prior'а `λ_{ℓ_2}` влияет на внутреннюю структуру `(λ_u, α)` и на скоринговое качество модели.
