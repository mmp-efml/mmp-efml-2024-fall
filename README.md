# Введение в эффективные системы машинного обучения

Этот репозиторий содержит материалы для курса по эффективному глубокому обучению. Курс предназначен для бакалавров и магистров и может быть зачтен как дисциплина по выбору.

На курсе вы сможете узнать:
- Какими способами можно сжимать модельки
- Как автоматически находить архитектуры
- Как работать с не самыми стандартными доменами данных
- Об особенностях современных генеративных моделей

## Описание курса

Курс "Эффективное глубокое обучение" охватывает основные концепции и методы глубокого обучения, а также рассматривает современные подходы и техники для повышения эффективности моделей. Курс включает в себя теоретические лекции, практические задания и проекты.

## Расписание занятий

- Время: пятница 10:30-12:05
- Аудитория: 507 (ВМК / 2ГУМ)
- Начало: 20 сен 2024


## Правила оценивания

В курсе планируются два домашних задания:
- Методы сжатия нейронных сетей
- Работа с нетипичными для глубокого обучения данными

За каждое домашнее задание можно получить зачет или незачет. Если зачтены оба домашних задания, то за курс ставится оценка "отлично". Если одно, то оценка "хорошо". В остальных случаях курс не считается сданным.

## Содержание курса

Записи занятий на YouTube [в плейлисте](https://www.youtube.com/playlist?list=PLmqlXGZVoej1sAsUNFIDoGleiTtb4Y1Pa).

| № | Дата | Занятие  | Материалы |
|----|-----|----------|-----------|
| 1  | 20.09.2024 | Дистилляция знаний| [Презентация](presentations/lec1_knowledge_distillation.pdf) [Ноутбук](notebooks/sem1_knowledge_distillation.ipynb) |
| 2  | 04.10.2024 | Квантизация и спарсификация| [Презентация](presentations/lec2_quantization_sparsification.pdf) [Ноутбук](notebooks/sem2_quantization/sem2_quantization.ipynb) |
| 3  | 11.10.2024 | PEFT & NAS| [Презентация](presentations/lec3_peft_nas.pdf) [Ноутбук](notebooks/sem3_peft_nas.ipynb) |
| 4  | 18.10.2024 | Временные ряды | [Конспект](presentations/lec4_time_series_analysis/notes.pdf) [Ноутбук](https://github.com/thuml/Time-Series-Library/blob/main/tutorial/TimesNet_tutorial.ipynb)
| 5  | 26.10.2024 | Рекомендательные системы | [Слайды](presentations/lec5_recommender_systems/slides.pdf) [Конспект](presentations/lec5_recommender_systems/notess.pdf) |
| 6  | 01.11.2024 | Текстовый поиск. RAG | [Слайды](presentations/lec6_textsearch_rag/slides.pdf) [Ноутбук](notebooks/sem6_textsearch_rag.ipynb) |
| 7  | 08.11.2024 | Обработка графов, графовые нейронные сети | [Слайды](presentations/lec7_gnn.pdf) [Пример работы с DGL](https://docs.dgl.ai/en/1.1.x/tutorials/blitz/1_introduction.html) |
| 8  | 15.11.2024 | Табличный DL | - |
| 9  | 22.11.2024 | Диффузионные модели, дискретное и непрерывное время | [Слайды](presentations/lec9_diffusion.pdf) |
| 10 | 29.11.2024 | Единое представление диффузии. Солверы ОДУ/СДУ | [Слайды](presentations/lec10_edm_solvers.pdf) [Ноутбук](notebooks/sem10_sampling.ipynb)|
| 11 | 06.12.2024 | Stable diffusion. Применение диффузии в качестве функции потерь | [Слайды](presentations/lec11_sd_sds_dmd.pdf) |

## Пререквизиты

Для успешного прохождения курса рекомендуется
- Иметь базовые знания в области программирования на Python и PyTorch
- Иметь представление о глубоком обучении (backpropagation, gradient optimization etc)
- Быть знакомым с устройством полносвязных, свёрточных, рекуррентных сетей и трансформеров

## Контакты

Если у вас есть вопросы или предложения, пожалуйста, свяжитесь с нами в тг: `@TrandeLik`, `@voorhs`, `@welmud`
