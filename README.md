# Fraud Detection Using Machine Learning

## 📌 Project Overview

This project implements an **end-to-end Fraud Detection system** using **Machine Learning** to identify fraudulent financial transactions. It simulates real-world banking and digital payment scenarios and focuses on detecting fraud early to minimize financial loss and improve customer trust.

The project covers the complete **data science lifecycle**—from data understanding and preprocessing to model deployment as an interactive web application.

---

## 🎯 Problem Statement

Financial institutions process millions of transactions every day. A small fraction of these transactions are fraudulent but can lead to significant financial loss and reputational damage. Traditional rule-based systems are not scalable and often fail to capture complex fraud patterns.

This project aims to build an **automated, data-driven fraud detection system** using machine learning.

---

## 🧠 Business Objective

* Detect fraudulent transactions accurately
* Reduce financial losses caused by fraud
* Minimize false alerts for genuine users
* Enable near real-time fraud detection

---

## 📊 Dataset Description

* Financial transaction dataset
* Each row represents one transaction
* Key features include:

  * Transaction amount
  * Sender and receiver balances (before & after)
  * Transaction type (TRANSFER, CASH-OUT, etc.)
  * Time step
* Target variable:

  * `isFraud = 1` → Fraudulent transaction
  * `isFraud = 0` → Legitimate transaction

The dataset closely resembles real-world digital payment behavior.

---

## 🔍 Exploratory Data Analysis (EDA)

Key insights from EDA:

* Fraud cases are extremely rare (highly imbalanced data)
* Most fraud occurs in **TRANSFER** and **CASH-OUT** transactions
* Fraud transactions often involve high amounts and sudden balance drops

These insights guided feature engineering and model selection.

---

## 🛠️ Data Preprocessing & Feature Engineering

* Handled categorical variables using encoding
* Scaled numerical features
* Removed irrelevant identifier columns
* Created engineered features such as **balance difference**

Feature engineering helped the model capture hidden fraud patterns.

---

## 🤖 Machine Learning Models

Models implemented:

* **Logistic Regression** – Baseline model
* **Random Forest Classifier** – Final model

### Final Model

* **Random Forest with SMOTE** for class imbalance handling
* Chosen for high recall and stable performance

---

## 📈 Model Evaluation

Evaluation metrics used:

* Precision
* Recall (prioritized to avoid missing fraud)
* F1-Score
* ROC-AUC

The final model achieves strong fraud detection capability with business-focused performance.

---

## 🚀 Deployment

The trained model is deployed as a **Streamlit web application**.

### Application Features:

* Real-time fraud prediction
* Adjustable fraud probability threshold
* Risk classification (High / Medium / Low)
* Feature importance visualization
* AI-assisted explanation of predictions
* Downloadable **PDF fraud report** for audit and compliance

---

## 🏗️ Project Structure

```
fraud_detection_project/
│
├── data/
│   └── Fraud_Analysis_Dataset.xlsx
│
├── notebooks/
│   └── fraud_detection_eda_model.ipynb
│
├── model/
│   ├── rf_smote_model.pkl
│   └── scaler.pkl
│
├── app/
│   └── main.py
│
├── requirements.txt
└── README.md
```

---

## ⚙️ How to Run the Project

1. Clone the repository
2. Create and activate a virtual environment
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit app:

   ```bash
   streamlit run app/main.py
   ```

---

## 📈 Results & Business Impact

* Improved detection of fraudulent transactions
* Reduced chances of missing fraud cases
* Faster and automated fraud screening
* Improved transparency through explainable AI

---

## ⚠️ Challenges Faced

* Highly imbalanced dataset
* Risk of missing fraud cases
* Need for explainable model decisions

### Solutions

* Used **SMOTE** to handle class imbalance
* Focused on **Recall** as a key metric
* Added feature importance and AI explanations

---

## 🔮 Future Enhancements

* Real-time transaction streaming
* Deep learning-based fraud detection
* Cloud deployment (AWS / Azure)
* Advanced explainability using SHAP
* Integration with banking APIs

---

## 👤 Author

**Your Name**
Aspiring Data Scientist | Machine Learning Enthusiast

---

## 📌 Note

This project is developed for **learning, demonstration, and interview preparation purposes** and simulates real-world fraud detection scenarios.
