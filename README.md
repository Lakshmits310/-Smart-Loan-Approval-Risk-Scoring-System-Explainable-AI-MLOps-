# **Smart Loan Approval & Risk Scoring System (Explainable AI + MLOps)**

### **Live Demo**
[👉 Click here to try the app]([https://your-app-url.streamlit.app](https://smart-loan-approval-risk-scoring-system-hjem2lirywhaki2ikh3mws.streamlit.app/))

## **Overview**
A **loan default prediction system** that uses **Explainable AI (XAI)** to help financial institutions and applicants understand the risk behind loan approvals.
* Predicts **default probability**
* Explains predictions using **SHAP**
* Provides **personalized recommendations**
* Deployed with **Streamlit Cloud** and **Docker**
* Integrated **CI/CD with GitHub Actions**

## **Key Features**
* **Risk Score Prediction** – Low/Medium/High risk with probability
* **Explainability (SHAP)** – Visual waterfall plots showing feature impact
* **Interactive Dashboard** – Gauge meter (green/yellow/red)
* **MLOps Workflow** – Dockerized, CI/CD pipeline with GitHub Actions
* **Real-Time Deployment** – Accessible via Streamlit Cloud

## **Tech Stack**
* **Frontend / Dashboard:** Streamlit, Plotly
* **Backend / Model:** Scikit-learn, XGBoost, SHAP
* **MLOps:** Docker, GitHub Actions (CI/CD)
* **Deployment:** Streamlit Cloud
* **Version Control:** Git, GitHub

## **Project Structure**
```bash
loan-risk-xai/
│
├── dashboard/                  # Streamlit dashboard app
│   └── app.py
│   └── models/                  # Saved model & preprocessing artifacts
|       ├── loan_risk_model.pkl
│       ├── scaler.pkl
│       └── feature_columns.pkl
│
├── tests/                      # Dummy tests for CI/CD
│   └── test_dummy.py
│
├── requirements.txt            # Project dependencies
├── Dockerfile                   # Docker build file
├── .github/workflows/ci.yml     # CI pipeline (GitHub Actions)
└── README.md                    # Project documentation
```

## **Implementation Steps**
### 1. **Data Preparation & Modeling**
* Preprocess dataset (encoding, scaling, train-test split)
* Train XGBoost / Logistic Regression
* Save artifacts (`model.pkl`, `scaler.pkl`, `feature_columns.pkl`)

### 2. **Explainability**
* Integrate SHAP for feature importance (local + global)

### 3. **Streamlit Dashboard**
* Collect applicant details (inputs)
* Predict risk probability
* Show gauge chart + SHAP waterfall
* Provide personalized recommendations

### 4. **MLOps**
* Dockerize the application
* Setup GitHub Actions CI pipeline (runs tests on push)
* Deploy on Streamlit Cloud (auto-pulls from GitHub)

## **Input Example**
```text
Age: 30
Annual Income: ₹500000
Loan Amount: ₹200000
Credit History Length: 5 years
Home Ownership: RENT
Loan Intent: EDUCATION
Loan Grade: B
Historical Default: N
```

## **Output Example**
* **Prediction:** High Risk – Probability: 16.5%
* **Gauge Chart:** Red zone
* **Context:** “Your profile risk is higher than 85% of applicants — proceed with caution.”
* **Recommendations:**
  * Consider reducing loan size
  * Improve credit history
  * Avoid multiple concurrent loans

## **How to Run Locally**
```bash
git clone https://github.com/<your-username>/loan-risk-xai.git
cd loan-risk-xai
pip install -r requirements.txt
streamlit run dashboard/app.py
```

## **Docker Run**
```bash
docker build -t loan-risk-xai .
docker run -p 8501:8501 loan-risk-xai
```

## **CI/CD Pipeline**
* Push changes → GitHub Actions runs tests → Auto-build
* Streamlit Cloud pulls latest build → Live app updates

