 Netflix Churn Analysis — Executive Dashboard

![Netflix Logo](https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg)

This project provides an **interactive dashboard** for analyzing and predicting customer churn in a Netflix-like dataset.  
It combines **data exploration, model prediction, and business impact analysis** into a single Streamlit app.

---

## 🔹 Features

- **Data Explorer**: Explore raw and cleaned customer datasets.  
- **Visual Insights**: View key charts, distributions, and model diagnostics (CV AUC scores, sensitivity heatmaps).  
- **Model Playground**: Predict churn for single customers (JSON) or batch predictions (CSV) using a trained Random Forest model.  
- **Business Impact**: Review ROI, cost-benefit analyses, and retention strategies.  
- **Deploy Ready**: Easily deploy the dashboard on Streamlit Cloud or other hosting platforms.

---

## 📁 Folder Structure

Netflix-Churn-Data-Analysis/
│── app/
│ └── app.py # Main Streamlit app
│── components/
│ └── utils.py # Helper functions
│ └── ui.py # UI rendering functions
│── visuals/
│ └── *.png / *.jpg / *.html # Generated charts and plots
│── models/
│ └── churn_pipeline_rf_v1.pkl # Trained Random Forest model
│── dataset.zip # Optional local dataset
│── requirements.txt # Python dependencies
│── README.md

yaml
Copy code

---

## ⚡ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/kimenjuivy/Netflix-Churn-Data-Analysis.git
cd Netflix-Churn-Data-Analysis/app
2. Create Python Environment
bash
Copy code
conda create -n data_env python=3.10
conda activate data_env
3. Install Dependencies
bash
Copy code
pip install -r ../requirements.txt
4. Run the App
bash
Copy code
streamlit run app.py
📊 Dataset
The dataset is not included in the repo due to size.

Download dataset.zip from the GitHub Releases.

The app automatically reads the dataset from the release if no local copy exists.

🔧 Usage
Home Page – Overview of the project and quick links.

Data Explorer – Preview raw datasets and view saved visualizations.

Model Playground – Perform single-row or batch churn predictions.

Insights – Explore model diagnostics and key saved visuals.

Business Impact – Review cost-benefit and retention analysis.

Deploy – Instructions for deploying the app on Streamlit Cloud.

🛠 Dependencies
Python 3.10+

pandas

numpy

scikit-learn

matplotlib, seaborn, plotly

streamlit

streamlit-components (for HTML plots)

All dependencies are listed in requirements.txt.

🚀 Deployment
Push the repository to GitHub (include app/, models/, visuals/, requirements.txt).

Connect the repo to Streamlit Cloud.

Set app/app.py as the main file.

Upload dataset.zip to GitHub Releases for automatic download by the app.

📈 Visuals
PNG/JPG charts: displayed with Streamlit.

HTML charts: displayed using st.components.v1.html.

CV AUC scores, cost-benefit tables, and heatmaps included in visuals/.

🔗 Links
GitHub Repository

GitHub Releases (dataset)

✨ Notes
Ensure models/churn_pipeline_rf_v1.pkl exists to use the Model Playground.

Keep large raw datasets out of the main branch; use GitHub Releases instead.

