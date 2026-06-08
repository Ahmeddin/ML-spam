import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_model():
    dataset_path = 'mail_data.csv'
    
    # 1. Ensure dataset exists before running
    if not os.path.exists(dataset_path):
        print(f"Error: '{dataset_path}' not found. Please download the dataset and place it in this directory.")
        return

    print("Loading dataset...")
    df = pd.read_csv(dataset_path)

    # 2. Data Cleaning: Drop missing values to prevent execution bugs
    df = df.dropna(subset=['text', 'label_num'])

    X = df['text']
    y = df['label_num'].astype(int)  # 0 = Ham, 1 = Spam

    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Vectorizing text data via TF-IDF...")
    # 4. Feature Extraction
    # ngram_range=(1,2) helps capture word pairs like "bank account" or "click here"
    vectorizer = TfidfVectorizer(stop_words='english', min_df=2, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("Training Logistic Regression Classifier...")
    # 5. Model Training (Handled class imbalance via balanced weights)
    model = LogisticRegression(class_weight='balanced', random_state=42)
    model.fit(X_train_tfidf, y_train)

    # 6. Evaluation Matrix (Crucial for university presentation)
    y_pred = model.predict(X_test_tfidf)
    print("\n" + "="*40)
    print("      MODEL EVALUATION METRICS")
    print("="*40)
    print(f"Accuracy Score: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Ham (Normal Mail)', 'Spam']))
    print("="*40)

    # 7. Serialize and Export
    print("\nExporting artifacts to disk...")
    with open('vectorizer.pkl', 'wb') as vec_file:
        pickle.dump(vectorizer, vec_file)
        
    with open('model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
        
    print("Artifacts successfully saved: 'vectorizer.pkl' and 'model.pkl'")

if __name__ == '__main__':
    train_model()