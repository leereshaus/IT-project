import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Conv1D, GlobalMaxPooling1D, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# 1. Загрузка данных
file_path = 'data/prep_dataset.csv'
data = pd.read_csv(file_path)  # Замените на путь к вашему файлу
texts = data['main_text'].values
labels = data['category'].values

# 2. Предобработка данных
# Преобразование меток в числовой формат
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)

X_train_val, X_test, y_train_val, y_test = train_test_split(texts, labels_encoded, test_size=0.2, stratify=labels_encoded, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=42)

# 3. Векторизация текста (для RNN и CNN) и EarlyStopping
max_length = 100  # Максимальная длина последовательности
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq = tokenizer.texts_to_sequences(X_val)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_padded = pad_sequences(X_train_seq, maxlen=max_length)
X_val_padded = pad_sequences(X_val_seq, maxlen=max_length)
X_test_padded = pad_sequences(X_test_seq, maxlen=max_length)

y_train_one_hot = to_categorical(y_train, num_classes=num_classes)
y_val_one_hot = to_categorical(y_val, num_classes=num_classes)
y_test_one_hot = to_categorical(y_test, num_classes=num_classes)

# Модель CNN
def create_cnn_model():
    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=max_length))
    model.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def train_model():
    early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

    cnn_model = create_cnn_model()
    history_cnn = cnn_model.fit(
        X_train_padded, y_train_one_hot,
        validation_data=(X_val_padded, y_val_one_hot),
        epochs=10,
        batch_size=32,
        callbacks=[early_stopping]
    )

    cnn_loss, cnn_accuracy = cnn_model.evaluate(X_test_padded, y_test_one_hot)
    print(f"Точность CNN: {cnn_accuracy:.4f}")

    # Сохранение весов модели
    joblib.dump(tokenizer, 'modelCNN.tokenizer.joblib')
    cnn_model.save_weights('modelCNN.weights.h5')  # Убедитесь, что путь правильный


# Запуск обучения, если этот файл запускается напрямую
if __name__ == "__main__":
    train_model()
