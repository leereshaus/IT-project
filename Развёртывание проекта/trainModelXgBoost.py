import pandas as pd
import numpy as np
import tensorflow as tf
import xgboost as xgb
import joblib

from keras.layers import Bidirectional

from sklearn.feature_extraction.text import TfidfVectorizer
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

# Извлечение признаков с помощью TF-IDF
vectorizer = TfidfVectorizer(max_features=2500) # Ограничиваем количество признаков
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(X_test)

# Модель XgBoost
model_xgb = xgb.XGBClassifier(objective='multi:softmax', num_class=num_classes, random_state=42)
model_xgb.fit(X_train_tfidf, y_train)
accuracy_xgb = model_xgb.score(X_test_tfidf, y_test)

print(accuracy_xgb)

# Сохранение модели и векторизатора
joblib.dump(model_xgb, 'model_xgboost.joblib')
joblib.dump(vectorizer, 'vectorizer_xgboost.joblib')
joblib.dump(label_encoder, 'label_encoder_xgboost.joblib')