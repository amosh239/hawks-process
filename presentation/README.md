# Предзащита: слайды

Файл `predefence_slides.md` — slides для 10-минутного предзащитного выступления, формат [Marp](https://marp.app).

## Сборка

### Через Marp CLI

```bash
# Установка (один раз)
npm install -g @marp-team/marp-cli

# PDF
marp predefence_slides.md --pdf --allow-local-files

# HTML
marp predefence_slides.md --html --allow-local-files

# PowerPoint (.pptx)
marp predefence_slides.md --pptx --allow-local-files
```

`--allow-local-files` нужен из-за ссылок на PNG в `../diploma/reports/...`.

### Через VS Code

Установить расширение **Marp for VS Code** → открыть `.md` → нажать иконку с глазом сверху справа → "Open Preview to the Side". Экспорт через Marp menu.

## Структура (16 слайдов)

| # | Слайд | Время |
| ---: | --- | ---: |
| 1 | Title | 10s |
| 2 | Цель работы | 30s |
| 3 | Что предсказываем | 30s |
| 4 | Hawkes basics | 60s |
| 5 | Данные и метрики | 30s |
| 6 | Статистика панели | 45s |
| 7 | New Year effect | 20s |
| 8 | Лестница базовых моделей | 45s |
| 9 | Personalized GP — EB | 60s |
| 10 | Feature correlation | 30s |
| 11 | Scaled-baseline Hawkes | 60s |
| 12 | GBDT и trade-off | 45s |
| 13 | Лестница (гл.9) | 45s |
| 14 | Переход к другим train-test | 60s |
| 15 | Pooled & Joint Hawkes | 60s |
| 16 | Train-length scan (гл.11) | 60s |
| 17 | Hawkes-параметры от n | 60s |
| 18 | Выводы | 30s |

Итого ~12 минут с запасом — для строго `10` минут стоит сократить детали в 9, 11, 14, 15.

Speaker notes — в HTML-комментариях, видны в presenter view (Marp CLI с `--preview` или VS Code Marp-extension).
