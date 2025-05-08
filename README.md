# 👁️ Diabetic Retinopathy Detection App

A web-based application that uses deep learning to detect **Diabetic Retinopathy (DR)** stages from retinal fundus images. It also generates **Grad-CAM heatmaps** to visualize the areas the model focused on while making its prediction.

---

## 🚀 Live Demo

👉 [Click here to try the app](https://your-username-your-repo-name.streamlit.app)  
> _(Hosted using Streamlit Community Cloud)_

---

## 🧠 How It Works

- Model: `EfficientNetB0` trained on the APTOS 2019 dataset
- Classification: 5-stage DR severity (No DR → Proliferative DR)
- Explainability: Grad-CAM heatmaps highlight retinal regions used in the prediction

---

## 📁 Files in This Repo

| File | Purpose |
|------|---------|
| `app.py` | Main Streamlit app |
| `dr_model.h5` | Trained CNN model |
| `requirements.txt` | List of required Python packages |
| `logo.png` | (Optional) Logo/banner image |
| `README.md` | This file |

---

## 🧪 DR Stages (Model Output)

| Class Index | DR Stage |
|-------------|----------|
| 0 | No DR |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |
| 4 | Proliferative DR |

---

## 🔧 Setup Instructions (for local run)

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
streamlit run app.py
