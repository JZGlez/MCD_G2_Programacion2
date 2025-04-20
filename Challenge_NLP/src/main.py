
import pandas as pd
import mlflow
import nltk
import spacy
import re
from datetime import datetime
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pysentimiento import create_analyzer

# Download necessary resources
print("\033[93m Downloading NLTK Stopwords library \033[0m")
nltk.download('stopwords')
print("\033[92m NLTK Stopwords library download... OK \033[0m")
print("")

print("\033[93m Downloading NLTK Punctuation library \033[0m")
nltk.download('punkt')
print("\033[92m NLTK Punkt library download... OK \033[0m")
print("")

print("\033[93m Downloading Spacy English model \033[0m")
spacy_en = spacy.load("en_core_web_sm")
print("\033[92m Spacy English Language model download... OK \033[0m")
print("")

# Text cleaning and preprocessing methods
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-záéíóúñü\s]", "", text)
    return text

def preprocess_texts(df, text_col='headline'):
    print("\033[93m Pre-processing started \033[0m")

    def process_texts(df, text_col, nlp, lang):
        print("\033[93m Processing started \033[0m")
        stop_words = set(stopwords.words(lang))
        cleaned_texts = []
        print("\033[93m Cleaning in process \033[0m")
        for doc in df[text_col]:
            cleaned = clean_text(doc)
            tokens = [token.lemma_ for token in nlp(cleaned) if token.lemma_ not in stop_words and token.is_alpha]
            cleaned_texts.append(' '.join(tokens))
        print("\033[92m Text cleaning... Done \033[0m")
        print("\033[92m Processing complete. \033[0m")
        return cleaned_texts

    df["processed"] = process_texts(df, text_col, spacy_en, 'english')
    print("\033[92m Pre-processing... Done \033[0m")
    print("")
    return df

# N-gramas method
def ngram_distribution(texts, n=2):
    print("\033[93m N-gram process started \033[0m")
    vectorizer = CountVectorizer(ngram_range=(n, n))
    X = vectorizer.fit_transform(texts)
    ngram_freq = zip(vectorizer.get_feature_names_out(), X.sum(axis=0).tolist()[0])
    print("\033[92m N-gram process ended.... Completed \033[0m")
    print("")
    return sorted(ngram_freq, key=lambda x: x[1], reverse=True)

# Classfying method
def classify_texts(texts, labels, lang='en'):
    print("\033[93m Classification process started \033[0m")

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    mlflow.log_metric(f"accuracy_{lang}", acc)
    mlflow.log_text(report, f"classification_report_{lang}.txt")
    print("\033[92m Classification Process .... Completed \033[0m")
    print("\033[94m Classification Model Accuracy:  \033[0m", acc)
    print("")

    return clf, vectorizer

# Sentiment Analysis method
def sentiment_analysis(texts, lang='en'):
    results = []
    if lang == 'en':
        print("\033[93m Sentiment Analysis.... started \033[0m")
        analyzer = SentimentIntensityAnalyzer()
        for text in texts:
            scores = analyzer.polarity_scores(text)
            results.append(scores)
        print("\033[92m Sentiment Analysis Process .... Completed \033[0m")
        print("")
    return results

# General Pipeline
def run_nlp_pipeline(df, label_col='recommend', end_date=None):
    with mlflow.start_run(run_name="Glassdoor NLP Pipeline"):
        mlflow.log_param("end_date", end_date or datetime.now().isoformat())

        df_en = preprocess_texts(df)

        if label_col in df_en:
            classify_texts(df_en["processed"], df_en[label_col], lang='en')
            sentiments_en = sentiment_analysis(df_en["processed"], lang='en')
            mlflow.log_metric("num_english_reviews", len(df_en))

        bigrams_en = ngram_distribution(df_en["processed"], n=2)
        mlflow.log_text(str(bigrams_en[:20]), "top_bigrams_en.txt")
        print("\033[94m Pipeline complete and registered in MLFlow\033[0m")
