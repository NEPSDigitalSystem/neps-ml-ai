import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


class EmotionDetectionModel:
    """
    Emotion Detection Model

    Predicts emotional states from text using:
    - TF-IDF Vectorization
    - Logistic Regression
    """

    def __init__(self):

        self.vectorizer = TfidfVectorizer(
            max_features=15000,
            ngram_range=(1, 2)
        )

        self.model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )

    # ---------------------------------
    # TRAIN MODEL
    # ---------------------------------

    def train(self, X, y):

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )

        X_train_tfidf = self.vectorizer.fit_transform(X_train)

        X_test_tfidf = self.vectorizer.transform(X_test)

        self.model.fit(
            X_train_tfidf,
            y_train
        )

        y_pred = self.model.predict(
            X_test_tfidf
        )

        print(
            classification_report(
                y_test,
                y_pred
            )
        )

        return X_test_tfidf, y_test, y_pred

    # ---------------------------------
    # PREDICT
    # ---------------------------------

    def predict(self, text_list):

        X = self.vectorizer.transform(
            text_list
        )

        return self.model.predict(X)

    # ---------------------------------
    # PREDICT PROBABILITIES
    # ---------------------------------

    def predict_proba(self, text_list):

        X = self.vectorizer.transform(
            text_list
        )

        return self.model.predict_proba(X)

    # ---------------------------------
    # FEATURE IMPORTANCE
    # ---------------------------------

    def get_feature_importance(self):

        feature_names = self.vectorizer.get_feature_names_out()

        emotion_features = {}

        for i, emotion in enumerate(
            self.model.classes_
        ):

            top_idx = self.model.coef_[i].argsort()[-15:]

            emotion_features[emotion] = [

                feature_names[j]

                for j in top_idx
            ]

        return emotion_features

    # ---------------------------------
    # SAVE MODEL
    # ---------------------------------

    def save(self, path="models"):

        os.makedirs(
            path,
            exist_ok=True
        )

        joblib.dump(
            self.model,
            f"{path}/emotion_model.pkl"
        )

        joblib.dump(
            self.vectorizer,
            f"{path}/emotion_vectorizer.pkl"
        )

    # ---------------------------------
    # LOAD MODEL
    # ---------------------------------

    def load(self, path="models"):

        self.model = joblib.load(
            f"{path}/emotion_model.pkl"
        )

        self.vectorizer = joblib.load(
            f"{path}/emotion_vectorizer.pkl"
        )