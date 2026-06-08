# Intelligence-Driven Email Spam Classifier

An automated machine learning solution to classify email communications into **Ham** (legitimate) and **Spam** (unsolicited). This project leverages Natural Language Processing (NLP) and Logistic Regression to ensure high-accuracy filtering.

## 🚀 Features
- **Real-time Classification:** Built with Streamlit for instant feedback.
- **Robust NLP Pipeline:** Utilizes TF-IDF Vectorization (`ngram_range=(1,2)`) for precise linguistic analysis.
- **High Accuracy:** Model achieves ~97% accuracy on test datasets.
- **Model Serialization:** Uses `pickle` to store artifacts, ensuring fast inference without retraining.

## 🛠 Tech Stack
- **Languages:** Python
- **Libraries:** Scikit-Learn, Pandas, NumPy, Streamlit
- **Development:** VS Code, Jupyter Notebooks

## 📂 Project Structure
- `app.py`: Main Streamlit application entry point.
- `Create.py`: Training script to generate `model.pkl` and `vectorizer.pkl`.
- `mail_data.csv`: Source dataset for model training.

## 💻 How to Run Locally
1. **Clone the repository:**
```bash
   git clone [https://github.com/your-username/ML-spam.git](https://github.com/Ahmeddin/ML-spam.git)
   cd ML-spam