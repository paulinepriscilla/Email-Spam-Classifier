EMAIL SPAM CLASSIFIER

Project Overview

The Email Spam Classifier is a Machine Learning project that classifies emails as **Spam** or **Not Spam (Ham)** based on their content.

The project uses Natural Language Processing (NLP) techniques to clean email text and convert it into numerical features before training a Machine Learning model.

This project is developed as part of an AI & Machine Learning Internship.

Objective

The main objective of this project is to:

- Detect spam emails automatically.
- Learn text preprocessing using NLP.
- Train a Machine Learning classification model.
- Evaluate model performance.
- Predict whether an email is spam or not.

Dataset

The dataset used in this project is the **SMS Spam Collection Dataset**.

It contains two columns:

- **Label** – Spam or Ham (Not Spam)
- **Message** – Email/SMS text

Labels

- Ham (Not Spam)
- Spam

Technologies Used

- Python
- Google Colab / Jupyter Notebook
- Pandas
- NumPy
- NLTK
- Scikit-learn
- TF-IDF Vectorizer

Machine Learning Model

The following Machine Learning model was used:

- Logistic Regression

The model classifies emails as Spam or Not Spam based on the processed text.

Data Preprocessing

The following preprocessing steps were performed:

- Removed unwanted columns
- Converted text to lowercase
- Removed URLs
- Removed punctuation
- Removed numbers
- Removed stop words
- Applied stemming
- Converted text into numerical features using TF-IDF Vectorization

Data Analysis

The project includes:

- Dataset information
- Spam vs Ham distribution
- Text preprocessing
- Model accuracy
- Classification report

Project Workflow

1. Load the dataset
2. Clean the email text
3. Perform text preprocessing
4. Convert text into TF-IDF features
5. Split the dataset
6. Train the Logistic Regression model
7. Evaluate model performance
8. Predict whether a new email is Spam or Not Spam

Output

The project displays:

- Dataset information
- Model accuracy
- Classification Report
- Spam/Not Spam prediction
- Prediction confidence

Project Files


Email-Spam-Classifier/
│
├── spam.csv
├── email_spam_classifier.py
├── README.md
└── requirements.txt


How to Run

1. Download the Spam dataset.
2. Open Google Colab or Jupyter Notebook.
3. Upload the dataset.
4. Run all the code cells.
5. Enter an email message.
6. The model predicts whether the email is Spam or Not Spam.

Future Improvements

- Improve model accuracy using advanced Machine Learning models.
- Build a web application using Streamlit.
- Add support for detecting phishing emails.
- Deploy the model as a web application.

Author

Pauline Priscilla C

AI & Machine Learning Internship Project

Acknowledgement

This project is developed for educational and internship purposes using Machine Learning and Natural Language Processing (NLP). The dataset is used for learning spam detection and text classification techniques.
