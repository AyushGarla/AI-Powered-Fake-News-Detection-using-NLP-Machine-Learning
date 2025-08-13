# 📰 Fake News Detection using NLP & Machine Learning

## 📌 Project Overview
This project implements an AI-powered fake news classifier using **Natural Language Processing (NLP)** and **Machine Learning**.  
It preprocesses text data, vectorizes it using **TF-IDF**, and trains models to classify news as **real (1)** or **fake (0)**.  
The final model, **Linear SVM**, achieves over **99% accuracy** on test data.

---

## 🎯 Objective
To build an accurate and efficient fake news detection system that can process raw news data and predict whether it is real or fake.

---

## 🛠️ Technologies & Tools
- **Python Libraries:** Pandas, Scikit-learn, NLTK, Regex
- **ML Models:** Linear SVM, Logistic Regression
- **Vectorization:** TF-IDF
- **Other:** Jupyter Notebook, Joblib for model saving/loading

---

## 📂 Dataset
- **Training Dataset:** `data.csv`
- **Validation Dataset:** `validation_data.csv`
- **Columns in Dataset:**
  - `label`: 0 = fake, 1 = real (in validation file, 2 is placeholder)
  - `title`: News headline
  - `text`: Full news content
  - `subject`: Topic of news
  - `date`: Publication date

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/fake-news-detection.git
cd fake-news-detection

Install Required Libraries
bash
Copy
Edit
pip install pandas scikit-learn nltk joblib
3️⃣ Download NLTK Data
In Python, run:

python
Copy
Edit
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
🚀 How to Run
1. Train the Model
Run the training script:

bash
Copy
Edit
python train_model.py
This will:

Load data.csv

Preprocess text (lowercase, remove special characters, stopwords, lemmatization)

Vectorize using TF-IDF

Train and evaluate Linear SVM and Logistic Regression

Save the best model (fake_news_svm_model.pkl) and vectorizer (tfidf_vectorizer.pkl)

2. Predict on Validation Data
Run the prediction script:

bash
Copy
Edit
python predict_validation.py
This will:

Load validation_data.csv

Apply the same preprocessing steps

Use the saved SVM model to predict labels

Replace label column values (2 → 0 or 1)

Save output as validation_data_predicted.csv

📊 Results
Linear SVM Accuracy: 99.37%

Logistic Regression Accuracy: 98.33%

Model successfully predicts fake/real news for unseen data.

📌 Future Enhancements
Use Transformer-based models (BERT, RoBERTa) for better contextual accuracy

Deploy as a real-time API or web application

Continuously update dataset for evolving fake news patterns

👨‍💻 Author
Ayush Garla
LinkedIn: https://www.linkedin.com/in/ayush-garla
GitHub: https://github.com/AyushGarla