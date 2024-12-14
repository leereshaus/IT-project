import numpy as np
import joblib
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense

# Загрузка токенизатора и модели CNN
tokenizer_cnn = joblib.load('modelCNN.tokenizer.joblib')  # Убедитесь, что используете правильный токенизатор
max_length = 100  # Убедитесь, что это соответствует обучению модели

# Функция для создания модели CNN
def create_cnn_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=len(tokenizer_cnn.word_index) + 1, output_dim=128, input_length=max_length))
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=5, activation='relu'))
    model.add(tf.keras.layers.GlobalMaxPooling1D())
    model.add(tf.keras.layers.Dense(9, activation='softmax'))  # Замените на правильное количество классов
    return model

# Загрузка модели CNN
cnn_model = create_cnn_model()
cnn_model.build((None, max_length))  # Инициализация модели с размером входных данных
cnn_model.load_weights('modelCNN.weights.h5')

# Загрузка токенизатора и модели GRU
tokenizer_gru = joblib.load('modelGRU.tokenizer.joblib')

# Функция для создания модели GRU
def create_gru_model():
    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer_gru.word_index) + 1, output_dim=128, input_length=max_length))
    model.add(GRU(64))  # Используем GRU
    model.add(Dense(9, activation='softmax'))  # Замените на правильное количество классов
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Загрузка модели GRU
gru_model = create_gru_model()
gru_model.build((None, max_length))  # Инициализация модели с размером входных данных
gru_model.load_weights('modelGRU.weights.h5')

# Загрузка модели XGBoost и векторизатора
model_xgb = joblib.load('model_xgboost.joblib')
vectorizer_xgb = joblib.load('vectorizer_xgboost.joblib')
label_encoder_xgb = joblib.load('label_encoder_xgboost.joblib')

# Загрузка модели SVM и векторизатора
model_svm = joblib.load('model_svm.joblib')
vectorizer_svm = joblib.load('vectorizer_svm.joblib')
label_encoder_svm = joblib.load('label_encoder_svm.joblib')

app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode("utf-8")

    # Векторизация текста для нейронных моделей
    sequences_cnn = tokenizer_cnn.texts_to_sequences([text])
    padded_cnn = pad_sequences(sequences_cnn, maxlen=max_length)

    # Предсказание от модели CNN
    predictions = {}
    cnn_prediction = cnn_model.predict(padded_cnn)
    cnn_predicted_class = cnn_prediction.argmax()  # Класс с максимальной вероятностью
    predictions["CNN"] = int(cnn_predicted_class)

    # Векторизация текста для модели GRU
    sequences_gru = tokenizer_gru.texts_to_sequences([text])
    padded_gru = pad_sequences(sequences_gru, maxlen=max_length)

    # Предсказание от модели GRU
    gru_prediction = gru_model.predict(padded_gru)
    gru_predicted_class = gru_prediction.argmax()  # Класс с максимальной вероятностью
    predictions["GRU"] = int(gru_predicted_class)

    # Векторизация текста для модели XGBoost
    xgb_input = vectorizer_xgb.transform([text])
    xgb_prediction = model_xgb.predict(xgb_input)
    predictions["XgBoost"] = int(xgb_prediction[0])  # Добавляем предсказание XGBoost

    # Векторизация текста для модели SVM
    svm_input = vectorizer_svm.transform([text])
    svm_prediction = model_svm.predict(svm_input)  # Исправлено: используем svm_input вместо xgb_input
    predictions["SVM"] = int(svm_prediction[0])  # Добавляем предсказание SVM

    return JSONResponse(content=predictions)

# Запуск приложения
# Для запуска приложения используйте команду: uvicorn app:app --reload