from datasets import load_dataset
import pandas as pd
import nltk
import re
import string
import joblib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import seaborn as sns
import matplotlib.pyplot as plt


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

print("STEP 1: LOADING DATASET")
dataset = load_dataset("banking77")
train_df = pd.DataFrame(dataset["train"])
test_df = pd.DataFrame(dataset["test"])

print("Dataset Loaded Successfully!")
print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

label_names = dataset["train"].features["label"].names
train_df["label_name"] = train_df["label"].apply(lambda x: label_names[x])
test_df["label_name"] = test_df["label"].apply(lambda x: label_names[x])


print("STEP 2: DATA CLEANING")

def perform_data_cleaning(df, text_column='text', verbose=True):
    if verbose:
        print("🔹 Starting Data Cleaning...")
    
    initial_rows = df.shape[0]
    
    df = df.dropna(subset=[text_column])
    df = df.drop_duplicates(subset=[text_column], keep='first')
    df = df[df[text_column].str.len() > 3]
    
    def normalize(text):
        text = str(text).lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = " ".join(text.split())
        return text
    
    df[text_column] = df[text_column].apply(normalize)
    final_rows = df.shape[0]
    
    if verbose:
        print(f"✅ Cleaning Done! Removed {initial_rows - final_rows} rows.")
        print(f"📊 Final shape: {df.shape}")
    
    return df

train_df = perform_data_cleaning(train_df)
test_df = perform_data_cleaning(test_df)

print("\n🔍 Sample of cleaned text:")
print(train_df['text'].head())

# STEP 3: PREPROCESSING
print("STEP 3: TEXT PREPROCESSING")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):

    tokens = word_tokenize(text)
    
    tokens = [word for word in tokens if word not in stop_words]
    
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return " ".join(tokens)

print("Applying preprocessing to train data...")
train_df['processed_text'] = train_df['text'].apply(preprocess_text)

print("Applying preprocessing to test data...")
test_df['processed_text'] = test_df['text'].apply(preprocess_text)

print("✅ Preprocessing complete!")
print(f"Sample: {train_df['processed_text'].iloc[0][:100]}...")


print("STEP 4: FEATURE ENGINEERING")

X_train_text = train_df['processed_text']
y_train = train_df['label']
X_test_text = test_df['processed_text']
y_test = test_df['label']

print("\n🔹 Applying CountVectorizer...")
count_vectorizer = CountVectorizer(
    stop_words='english',
    ngram_range=(1, 2),
    max_features=5000
)
X_train_count = count_vectorizer.fit_transform(X_train_text)
X_test_count = count_vectorizer.transform(X_test_text)
print(f"CountVectorizer - Train shape: {X_train_count.shape}")

print("\n🔹 Applying TF-IDF...")
tfidf_vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2),
    max_features=5000
)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
X_test_tfidf = tfidf_vectorizer.transform(X_test_text)
print(f"TF-IDF - Train shape: {X_train_tfidf.shape}")

X_train_final = X_train_tfidf
X_test_final = X_test_tfidf

joblib.dump(X_train_final, 'X_train.pkl')
joblib.dump(X_test_final, 'X_test.pkl')
joblib.dump(y_train, 'y_train.pkl')
joblib.dump(y_test, 'y_test.pkl')
joblib.dump(tfidf_vectorizer, 'vectorizer.pkl')
joblib.dump(count_vectorizer, 'count_vectorizer.pkl')

print("✅ All features saved!")

print("\n" + "="*50)
print("STEP 5: MODEL TRAINING")
print("="*50)

print("🚀 Training Logistic Regression model...")

model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    C=1.0
)

model.fit(X_train_final, y_train)

joblib.dump(model, 'banking_assistant_model.pkl')
print("✅ Model saved as 'banking_assistant_model.pkl'")
print("\n📊 Evaluating model on Test Data...")
y_pred = model.predict(X_test_final)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\n📋 Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=label_names)) 

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, fmt='d', cmap='Blues', xticklabels=label_names[:10], yticklabels=label_names[:10])
plt.title('Confusion Matrix - Banking Intents')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("✅ Confusion matrix saved as 'confusion_matrix.png'")

with open('model_evaluation_report.txt', 'w', encoding='utf-8') as f:
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write("Classification Report (first 10 classes):\n")
    f.write(classification_report(y_test, y_pred, target_names=label_names))

print("✅ Evaluation report saved as 'model_evaluation_report.txt'")

