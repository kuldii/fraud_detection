# PaySim Fraud Detection App

A professional, production-ready machine learning app for predicting financial fraud using the PaySim dataset from Kaggle. Built with robust preprocessing, user-friendly feature mapping, multiple classification models, and a modern Gradio & Streamlit UI. All models and preprocessing objects are saved for easy deployment.

---

## 🚀 Features

- **Robust Preprocessing**: Categorical encoding, numeric scaling, user-friendly feature mapping
- **Multiple Classification Models**: Logistic Regression, Random Forest, XGBoost (with hyperparameter tuning)
- **Rich Visualizations & EDA**: Class distribution, histograms, boxplots, correlation heatmap, feature statistics
- **Interactive Gradio & Streamlit UI**: Model selection, grouped input fields (dropdowns, numbers), clear prediction output, modern layout
- **Deployment-Ready**: All models and preprocessing objects are saved for instant prediction in the app

---

## 🏗️ Project Structure

```
fraud_detection/
├── app.py                    # Gradio app for prediction (production-ready)
├── streamlit_app.py          # Streamlit app for prediction (production-ready)
├── fraud_detection_analysis.ipynb # Full EDA, modeling, and training notebook
├── models/
│   └── fraud_detection_artifacts.pkl # Trained models and preprocessing objects (joblib)
├── requirements.txt          # Python dependencies (for Gradio app)
├── environment.yml           # Conda environment (for Streamlit app)
└── README.md                 # Project documentation
```

---

## 📊 Data & Preprocessing

- **Dataset**: [PaySim Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1)
- **Preprocessing**:
  - User-friendly feature names from dataset metadata
  - Label encoding for categoricals
  - Standardization of numeric features
  - Feature statistics (min, max, mean) for UI defaults

---

## 🧠 Models

- **Logistic Regression** (with GridSearchCV)
- **Random Forest Classifier** (with GridSearchCV)
- **XGBoost Classifier** (with GridSearchCV)

All models are trained and saved for instant prediction in the app.

---

## 🖥️ Gradio & Streamlit Apps

- **Dropdowns and numbers** for all features (custom min/max for each)
- **Model selection** dropdown/radio
- **Prediction output**: Fraud/Not Fraud and class probabilities
- **Production config**: Ready for local or cloud deployment

---

## 📦 Installation & Setup

### 1. Clone the Repository
```bash
git clone <this-repo-url>
cd fraud_detection
```

### 2. Install Dependencies
- For Gradio app:
```bash
pip install -r requirements.txt
```
- For Streamlit app (recommended):
```bash
conda env create -f environment.yml
conda activate fraud-detection-paysim
```

### 3. (Optional) Train Models
- All models and preprocessing objects are pre-trained and saved in `models/`.
- To retrain, use the notebook `fraud_detection_analysis.ipynb` and re-export the models.

### 4. Run the App
- Gradio:
```bash
python app.py
```
- Streamlit:
```bash
streamlit run streamlit_app.py
```

---

## 🖥️ Usage

1. Open the app in your browser.
2. Input transaction features (all fields are labeled and grouped for clarity).
3. Select a classification model.
4. Click **Predict Fraud** to get the prediction and class probabilities.

---

## 📊 Visualizations & EDA
- See `fraud_detection_analysis.ipynb` for:
  - Class distribution
  - Feature statistics
  - Correlation heatmap
  - Histograms, boxplots, and more

---

## 📝 Model Details
- **Preprocessing**: LabelEncoder for categoricals, StandardScaler for numerics, user-friendly feature mapping.
- **Models**: LogisticRegression, RandomForestClassifier, XGBoostClassifier (all with GridSearchCV for tuning).
- **Feature Info**: All feature statistics (min, max, mean) are saved for robust UI defaults.

---

## 📁 File Descriptions
- `app.py`: Gradio app, loads models, handles prediction and UI.
- `streamlit_app.py`: Streamlit app, loads models, handles prediction and UI.
- `models/fraud_detection_artifacts.pkl`: Dictionary of trained models and preprocessing objects.
- `requirements.txt`: Python dependencies for Gradio app.
- `environment.yml`: Conda environment for Streamlit app.
- `fraud_detection_analysis.ipynb`: Full EDA, preprocessing, model training, and export.

---

## 🌐 Demo & Credits
- **Author**: Sandikha Rahardi (Kuldii Project)
- **Website**: https://kuldiiproject.com
- **Dataset**: [PaySim Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1)
- **UI**: [Gradio](https://gradio.app/), [Streamlit](https://streamlit.io/)
- **ML**: [Scikit-learn](https://scikit-learn.org/), [XGBoost](https://xgboost.readthedocs.io/)

---

For questions or contributions, please open an issue or pull request.
