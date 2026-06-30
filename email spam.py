# ==========================================
# EMAIL SPAM CLASSIFIER
# ==========================================

import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Download stopwords (first run only)
nltk.download('stopwords')

# ==========================================
# LOAD DATASET
# ==========================================

# Make sure spam.csv is in the same folder
df = pd.read_csv("spam.csv", encoding="latin-1")

# Keep only first 2 columns
df = df.iloc[:, :2]
df.columns = ["label", "message"]

# Convert labels
df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

# ==========================================
# TEXT PREPROCESSING
# ==========================================

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove punctuation
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    words = text.split()

    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

df["clean_message"] = df["message"].apply(preprocess_text)

# ==========================================
# FEATURE EXTRACTION
# ==========================================

vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(df["clean_message"])
y = df["label"]

# ==========================================
# SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# TRAIN MODEL
# ==========================================

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# ==========================================
# EVALUATE MODEL
# ==========================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n================================")
print("EMAIL SPAM CLASSIFIER")
print("================================")

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ==========================================
# TEST CUSTOM EMAILS
# ==========================================

while True:

    email = input("\nEnter Email Text (or type exit): ")

    if email.lower() == "exit":
        print("Program Closed.")
        break

    cleaned_email = preprocess_text(email)

    vector = vectorizer.transform([cleaned_email])

    prediction = model.predict(vector)[0]

    probability = model.predict_proba(vector).max()

    if prediction == 1:
        print(f"\nSPAM EMAIL ({probability*100:.2f}% confidence)")
    else:
        print(f"\nNOT SPAM ({probability*100:.2f}% confidence)")