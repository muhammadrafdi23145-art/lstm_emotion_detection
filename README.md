# Emotion Detection with LSTM

**Deep Learning | NLP | Recurrent Neural Networks**

## Project Overview
This project focuses on identifying human emotions (such as Joy, Sadness, Anger, Fear, etc.) in text data. Using **LSTM (Long Short-Term Memory)**, the model is able to process sequences of words and understand the context better than standard machine learning algorithms.

## Model Architecture
- **Embedding Layer:** Converts words into dense vectors of fixed size.
- **Bidirectional LSTM:** Allows the model to learn context from both past and future states in a sentence.
- **Dropout & Dense Layers:** Used for regularization and final classification.

## Tech Stack
* **Framework:** TensorFlow / Keras
* **Natural Language Processing:** NLTK, Regex, Tokenization.
* **Libraries:** Scikit-Learn, Pandas, Seaborn.

## Performance
The model demonstrates high accuracy across multiple emotion classes. You can refer to `lstm_performance.png` for the training curves and confusion matrix.

## Repository Structure
- `lstm_emotion_detection.py`: Training script.
- `lstm_emotion_model.h5`: Saved Keras model.
- `tokenizer.pkl`: Fitted tokenizer for text vectorization.
- `label_encoder.pkl`: Encoder for target labels.

## Dataset
https://drive.google.com/file/d/1tHZ0HI6bv8-wnPdQDm_ttzQgcuFBZKIA/view?usp=sharing

## Result
<img width="1552" height="590" alt="image" src="https://github.com/user-attachments/assets/005baef2-0ec7-4b37-af61-f439a44144a4" />

## Evaluation

                   precision    recall      f1-score

    accuracy                                1.00
    macro avg       0.9394      0.9372      0.9369
    weighted avg    0.9393      0.9372      0.9369
