# Archive

Скрипты, использовавшиеся в research-фазе (до v2.0 диплома) и не нужные
для воспроизведения **актуальных** глав 1–11.

Они оставлены в репозитории для git-blame и истории решений. Никакая глава
текущего диплома и никакой живой `replot_*` / `plot_*` скрипт не читает их
артефакты, поэтому удаление любого файла отсюда не сломает `diploma/*.md`.

## Грубая классификация

| Группа | Файлы |
| --- | --- |
| Старые Hawkes-варианты до выбора `(1,3)` half-life | `run_experimental_1_1_*`, `run_experimental_1_2_hawkes_hl1.py`, `run_experimental_1_additive_hawkes.py` |
| Точечная диагностика staged / joint MLE-разъезжаемости | `check_train_*`, `verify_pooled_hawkes_ch6.py`, `diagnose_staged_hawkes_optimum.py`, `run_joint_out_of_fold_diagnostic.py`, `run_hawkes_test_fit_symmetric.py`, `run_hawkes_fit_on_test.py` |
| Sidetrack 14d-эксперименты с personalized L2 / frozen EB / staged-on-raw | `run_personalized_*`, `run_blockwise_cv_frozen_ch6.py` |
| Sweep'ы регуляризации, не вошедшие в финальный диплом | `run_hawkes_alpha_l2_sweep.py`, `run_blockwise_cv_hawkes_reg_sweep.py` |
| New Year эксперимент | `run_new_year_experiment.py` |

## Если понадобится восстановить

Скрипты используют тот же `src/diploma_baselines/` API что и живые. Достаточно
вернуть файл в `scripts/` и при необходимости подправить импорты —
рефакторинги в `src/` могут переименовать публичные функции.
