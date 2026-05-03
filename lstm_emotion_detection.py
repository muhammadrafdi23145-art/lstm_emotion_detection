# ==================================================
# Deep Learning: Emotion Detection using LSTM
# ==================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

# 1. Load Dataset
# Ensure your dataset CSV is in the same directory
df = pd.read_csv('emotion_data.csv') # Adjust filename accordingly

# 2. Text Preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

df['text_clean'] = df['text'].apply(clean_text)

# Label Encoding
le = LabelEncoder()
df['label_enc'] = le.fit_transform(df['label'])
num_classes = len(le.classes_)

# 3. Tokenization & Padding
MAX_WORDS = 10000
MAX_LEN = 100

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
tokenizer.fit_on_texts(df['text_clean'])

sequences = tokenizer.texts_to_sequences(df['text_clean'])
padded = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')

# 4. Split Data
X_train, X_test, y_train, y_test = train_test_split(
    padded, df['label_enc'], test_size=0.2, random_state=42, stratify=df['label_enc']
)

# 5. Build LSTM Model
model = Sequential([
    Embedding(MAX_WORDS, 64, input_length=MAX_LEN),
    Bidirectional(LSTM(64, return_sequences=False)),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 6. Training with Early Stopping
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=20,
    validation_split=0.1,
    batch_size=32,
    callbacks=[early_stop]
)

# 7. Evaluation & Visualization
y_pred = np.argmax(model.predict(X_test), axis=1)

# Plot Results
fig, ax = plt.subplots(1, 2, figsize=(15, 5))

# Accuracy Curve
ax[0].plot(history.history['accuracy'], label='Train Accuracy')
ax[0].plot(history.history['val_accuracy'], label='Val Accuracy')
ax[0].set_title('Training vs Validation Accuracy')
ax[0].legend()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=le.classes_, yticklabels=le.classes_, ax=ax[1])
ax[1].set_title('Confusion Matrix')
plt.tight_layout()
plt.savefig("lstm_performance.png")
plt.show()

# 8. Save Model & Tokenizer
model.save("lstm_emotion_model.h5")
joblib.dump(tokenizer, "tokenizer.pkl")
joblib.dump(le, "label_encoder.pkl")
print("Model and assets saved successfully!")
