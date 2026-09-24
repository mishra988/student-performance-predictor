# 🎓 Student Academic Performance Prediction Engine

> 🚀 **Live Web App:** **[https://student-performance-predictor-88bc.onrender.com](https://student-performance-predictor-88bc.onrender.com)**

[![Live Demo](https://img.shields.io/badge/Live_Demo-student--performance--predictor-00C7B7?style=for-the-badge&logo=render&logoColor=white)](https://student-performance-predictor-88bc.onrender.com)

[![Python](https://img.shields.io/badge/Python-3.13+-3776AB?logo=python&logoColor=white)](#)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9.1-F7931E?logo=scikit-learn&logoColor=white)](#)
[![Pandas](https://img.shields.io/badge/Pandas-3.0.6-150458?logo=pandas&logoColor=white)](#)
[![NumPy](https://img.shields.io/badge/NumPy-2.5.3-013243?logo=numpy&logoColor=white)](#)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.11.2-11557c)](#)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.13.2-4c72b0)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An end-to-end Machine Learning intelligence engine that predicts student final exam performance, categorizes academic risk tiers, identifies individual behavioral friction points, and computes counterfactual "what-if" prescriptive improvement recommendations.

---

## 🌟 Key Features

* **🧠 Multi-Model Benchmarking:** Evaluates Random Forest, Gradient Boosting, Ridge Regression, Support Vector Regressors (SVR), and Ordinary Least Squares using 5-fold cross-validation.
* **🔬 Domain-Specific Feature Engineering:**
  * **Study Efficiency Ratio:** Ratio of self-study investment to cognitive rest baseline.
  * **Attendance Risk Penalty:** Discontinuous penalty factor for attendance dipping below the 75% threshold.
  * **Academic Momentum:** Prior semester performance amplified by tutoring interventions.
  * **Wellbeing Balance Index:** Ratio of sleep hours to subjective stress levels.
* **📊 Visual Analytics Suite:** Generates 5 publication-ready charts using **Matplotlib** and **Seaborn**:
  * Correlation Matrix Heatmap
  * Actual vs. Predicted Regression Scatter
  * Feature Importance Rankings (Gini / MDI)
  * Residual Error Normality Distribution
  * Academic Performance Tiers Boxplots
* **🎯 Prescriptive Recommendation Engine:** Simulates personalized score uplifts (e.g., *"Increasing study hours from 8h to 14h yields a projected +6.2 score increase"*).
* **🖥️ Interactive Local Dashboard:** Clean web interface with real-time sliders, instant inference, and visual analytics gallery.

---

## 🏆 Model Benchmark Leaderboard

Evaluated on an out-of-sample holdout test set (500 samples, 20%):

| Model | 5-Fold CV $R^2$ | Test $R^2$ | RMSE | MAE | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting Regressor** | **0.8565 (±0.008)** | **0.8898** | **3.360** | **2.538** | 🏆 **Champion** |
| **Support Vector Regressor (SVR)** | 0.8495 (±0.006) | 0.8732 | 3.604 | 2.657 | Runner-Up |
| **Random Forest Regressor** | 0.7944 (±0.009) | 0.8316 | 4.153 | 2.987 | Strong Non-Linear |
| **Ridge Regression** | 0.7355 (±0.016) | 0.7566 | 4.993 | 3.820 | Linear Regularized |
| **Linear Regression** | 0.7355 (±0.017) | 0.7566 | 4.993 | 3.817 | Baseline |

---

## 🏗️ Architecture & Pipeline

```mermaid
flowchart TD
    RawData["Raw Academic & Behavioral Data\n(Study, Attendance, Sleep, Stress, Tutoring)"]
    FE["Feature Engineering Engine\n(Study Efficiency, Attendance Risk, Academic Momentum)"]
    Pipe["Scikit-Learn ColumnTransformer\n(StandardScaler, OrdinalEncoder, OneHotEncoder)"]
    Model["Champion Gradient Boosting Regressor\n(R² = 0.8898)"]
    Output["Predictions & Prescriptive Insights"]
    
    RawData --> FE
    FE --> Pipe
    Pipe --> Model
    Model --> Output
    Output --> Score["Expected Final Score (0 - 100)"]
    Output --> Tier["Risk Category (Distinction / Merit / Pass / At-Risk)"]
    Output --> Recs["Counterfactual Uplift Simulations"]
```

---

## 📂 Project Structure

```text
student-performance-predictor/
├── requirements.txt         # Dependencies (numpy, pandas, scikit-learn, matplotlib, seaborn)
├── .gitignore               # Ignored environments and caches
├── README.md                # Project documentation
├── data/
│   ├── raw/
│   │   └── student_data.csv # 2,500 student behavioral records
│   └── processed/
├── models/
│   ├── student_performance_model.joblib # Champion Gradient Boosting model
│   ├── preprocessor.joblib              # Fitted ColumnTransformer
│   └── model_metadata.json              # Benchmark metrics and feature rankings
├── artifacts/
│   └── visualizations/
│       ├── correlation_heatmap.png      # Seaborn correlation matrix
│       ├── actual_vs_predicted.png      # Regression fit plot
│       ├── feature_importance.png       # Gini importance bar chart
│       ├── residuals_distribution.png   # Error normality histogram
│       └── academic_tiers_distribution.png # Boxplots across performance tiers
├── src/
│   ├── data_generator.py     # Synthetic data generation with educational logic
│   ├── data_preprocessing.py # Feature engineering & ColumnTransformer
│   ├── train.py              # Model benchmarking, CV, and model serialization
│   ├── evaluate.py           # Matplotlib & Seaborn visual analytics suite
│   ├── predict.py            # Prescriptive inference & recommendation engine
│   └── app.py                # Interactive web dashboard & HTTP server
└── tests/
    └── test_pipeline.py      # Automated unit tests
```

---

## 🚀 Quick Start Guide

### 1. Activate the Virtual Environment
From the project folder:
```powershell
cd C:\Users\ayush\.gemini\antigravity\scratch\student-performance-predictor
.\venv\Scripts\Activate.ps1
```

### 2. Run the Interactive Web Dashboard
Launch the dashboard to interact with sliders and view real-time score updates and visualizations:
```powershell
python src\app.py
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser!

### 3. Run Individual Pipeline Stages

* **Re-train & Benchmark Models:**
  ```powershell
  python src\train.py
  ```
* **Generate Fresh Seaborn / Matplotlib Visualizations:**
  ```powershell
  python src\evaluate.py
  ```
* **Run CLI Single-Student Prediction:**
  ```powershell
  python src\predict.py
  ```
* **Execute Unit Tests:**
  ```powershell
  python -m unittest discover -s tests -p "test_*.py"
  ```

---

## 📊 Sample Inference Output

```text
--- Student Academic Performance Prediction ---
Predicted Final Exam Score: 74.5/100
Academic Tier: [Good] Good Standing (Merit)
Risk Level: Low

Risk Factors:
 - Sub-optimal attendance (72.0%): Falling below 75% triggers significant academic penalties.
 - Sleep deprivation (5.5 hours/night) impairs memory consolidation.
 - High subjective stress level (8/10) degrades exam retention.

Actionable Uplift Simulations:
 -> Increase weekly study by +6 hours (to 16.5h): 78.4 (+3.9 pts)
 -> Increase class attendance to 95%: 85.3 (+10.8 pts)
 -> Attend 2 additional tutoring sessions/month: 79.5 (+5.0 pts)
```

---

## 📄 License
MIT License. Built by Ayush.
