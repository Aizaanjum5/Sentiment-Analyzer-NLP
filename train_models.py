import pandas as pd
import numpy as np
import joblib
import nltk
import re
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

# ========== Load Dataset ==========
df = pd.read_csv("reviews.csv")  # use your correct CSV name!

# ========== Cleaning Function ==========
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_text'] = df['review'].apply(clean_text)

# ========== Classical ML ==========
tfidf = TfidfVectorizer(max_features=5000)
bow = CountVectorizer(max_features=5000)

X_tfidf = tfidf.fit_transform(df['clean_text'])
X_bow = bow.fit_transform(df['clean_text'])

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['sentiment'])

X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
X_train_bow, X_test_bow, _, _ = train_test_split(X_bow, y, test_size=0.2, random_state=42)

log_reg = LogisticRegression(max_iter=500)
log_reg.fit(X_train_tfidf, y_train)

nb = MultinomialNB()
nb.fit(X_train_bow, y_train)

# ========== Deep Learning (LSTM) ==========
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(df['clean_text'])
sequences = tokenizer.texts_to_sequences(df['clean_text'])
padded = pad_sequences(sequences, maxlen=100)

X_train_pad, X_test_pad, y_train_pad, y_test_pad = train_test_split(padded, y, test_size=0.2, random_state=42)

vocab_size = 5000
embedding_dim = 64

lstm_model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=100),
    LSTM(64),
    Dense(32, activation='relu'),
    Dense(len(np.unique(y)), activation='softmax')
])

lstm_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
lstm_model.fit(X_train_pad, y_train_pad, epochs=3, batch_size=32, validation_split=0.2)

# ========== Save Artifacts ==========
joblib.dump(tfidf, "saved_models/tfidf_vectorizer.joblib")
joblib.dump(log_reg, "saved_models/log_reg_model.joblib")

joblib.dump(bow, "saved_models/bow_vectorizer.joblib")
joblib.dump(nb, "saved_models/nb_model.joblib")

joblib.dump(tokenizer, "saved_models/tokenizer.joblib")
joblib.dump(label_encoder, "saved_models/label_encoder.joblib")

lstm_model.save("saved_models/lstm_model.h5")

print("Models Trained and Saved Successfully!")
