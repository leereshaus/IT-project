# IT-project

## Тема: Категоризация новостей с помощью чат-бота в Telegram

### Назначение ролей
| ФИ  | Роль |
| ------------- | ------------- |
| Афанасьев Денис  |   |
| Боттаева Амина  |   |
| Гусева Софья  |   |
| Склезнёва Ксения  |   |

### Постановка задачи
В ходе выполнения проекта командой было реализовано обучение моделей нейронных сетей и алгоритмов машинного обучения, было развернуто веб-приложение, в котором модели классифицируют подаваемые на вход новости в форме строки. Вывод программы приложения реализован в формате:
```
{
    “CNN”: “Class_CNN”,
    “GRU”: “Class_GRU”,
    “XgBoost”: “Class_XgB”,
    “SVM”: Class_SVM
}
```

### Данные

Датасет, составленный из новостей с сайта Lenta.ru за 2020 год. Код программы для парсинга новостей с сайта лежит в файле ```Разработка\Parser_news_LENTA.ipynb```
Все новости разбиты на 9 тем и пронумерованы для более удобной классификации:
* economy: 0
* sports: 1
* society: 2
* life: 3
* entertainment: 4
* technology: 5
* science: 6
* russia: 7
* history: 8

Предобработка получившегося датасета, включающая в себя лемматизацию/стемминг текста и создание разметки на основе тегов лежит в файле ```Разработка\data_preprocessing.ipynb```


### Решение задачи
#### Методы, используемы для предобработки текста:
* Приведение текста к нижнему регистру
* Удаление специальных символов и цифр
* Токенизация
* Удаление стоп-слов
* Приведение к нормальной форме
* Объединение токенов обратно в строку
#### Обученные модели:
* CNN (Свёрточные нейронные сети)
* RNN (Реккурентные нейронные сети)
* GRU (Gated Recurrent Units)
* BiLSTM (Двунаправленная LSTM)
* XgBoost (Градиентный спуск)
* SVM (Метод опорных векторов)

Обучение моделей нейронных сетей RNN, CNN, Bert(черновой вариант) лежит в файле ```Разработка\models.ipynb```.
Обучение моделей нейронных сетей GRU, BiLSTM и алгоритмов машинного обучения XgBoost и SVM лежит в файле ```Разработка\models_v2.ipynb```.


### Анализ полученных результатов

Сравнение обученных моделей по точности обучения на тренировочных данных:
| Модель  | RNN | CNN | GRU | BiLSTM | XgBoost | SVM |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| test data  | 0.8164 | 0.8354  | 0.8355  | 0.8161  | 0.8649  | 0.8801  |

### Развертывание проекта в веб-приложение

На основе полученных точностей обучения моделей и алгоритмов машинного обучения на тренировочных данных для развертки проекта в веб-сервис было принято решение использовать следующие модели:
* CNN
* GRU
* XgBoost
* SVM

Для развертки проекта для каждой модели были созданы отдельные скрипты, лежащие в файлах: ```Развёртывание проекта\trainModel.py*```, где * - название используемой модели. В коде также была реализована запись:
* Запись архитектуры и весов в файл ```model*.weights.h5``` для нейронных сетей CNN и GRU
* Запись обученного алгоритма в файл ```model_*.joblib ``` для алгоритмов машинного обучения XgBoost и SVM
* Запись векторизаторов в файл ```vectorizer_*.joblib``` для алгоритмов машинного обучения XgBoost и SVM
* Запись кодировок меток в файл ```label_encoder_*.joblib``` для алгоритмов машинного обучения XgBoost и SVM

Сам код, который использует созданные файлы в процессе компиляции кодов обучения моделей, находится в файле ```Развёртывание проекта\app.py*```.

#### Запуск приложения
Для запуска приложения необходимо скомпелировать сначала все файлы с обучением моделей, убедиться, что были созданы новые файлы с архитектурой, весами, алгоритмами, векторизаторами и кодировками для моделей, в которых это необходимо. Далее компелируется код программы самого приложения. После запуска приложения в терминале вводится команда:
```bash
uvicorn app:app --reload
```
После чего можно переходить по открывшейся ссылке и выбирать при запуске текстовый файл, в котором лежит новость, необходимая для классификации.

#### Наборы тестов
Все тестовые наборы представляют собой текст, записанный строкой в формате ```.txt```. Тесты лежат в соответствующей папке проекта.
