# Geological-logging
This project is aimed at building predictive models for lithology or hydrocarbon potential and classify rock types or estimate subsurface properties.It demonstrates how data science and machine learning can be applied to geological logging data to **predict bulk density (RHOB)** using well log features. The application includes a **Streamlit interface** and **LIME explainability** for model transparency.

## 📌 Project Objectives

- Predict **RHOB (bulk density)** using well log data (e.g., GR, DTC, NPHI).
- Build a regression model using **XGBoost**.
- Provide **model explanations** using **LIME** to interpret predictions.
- Deploy the solution via a **Streamlit web application**.


## 🧪 Features Used

The model uses the following well log features as inputs:

- `GR`: Gamma Ray
- `DEPTH_MD`: Measured Depth
- `NPHI`: Neutron Porosity
- `PEF`: Photoelectric Factor
- `DTC`: Compressional Slowness

Target variable: **RHOB** (Bulk Density)

## ⚙️ Technologies Used

- Python 3.8+
- XGBoost
- Scikit-learn
- Pandas, NumPy
- LIME (Local Interpretable Model-Agnostic Explanations)
- Matplotlib
- Streamlit (for deployment)

## 🚀 How to Run This Project

### 1. Clone the Repository

```bash
git clone git@github.com:JUDAH004/Geological-logging.git


## 📌 Project structure
geological-logging/
│
├── app.py                  # Streamlit app
├── xgb_model_rhob.pkl      # Trained model
├── requirements.txt        # Project dependencies
├── README.md               # Project description
└── data/
    └── Data/force2020_data_unsupervised_learning.csv      # Project data

You can view the Streamlit app in a browser.

  Local URL: http://localhost:8501

##📬 Contact
Author: Judah Samuel
Email: [judahsamuel.19@gmail.com]
LinkedIn: [linkedin.com/in/Judah-Samuel]
GitHub: [github.com/JUDAH004]