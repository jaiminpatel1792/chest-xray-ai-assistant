---
title: Chest X-ray AI Assistant
emoji: 🩻
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
---

# 🩺 Chest X-ray AI Assistant

An end-to-end **Deep Learning + MLOps** project that predicts chest diseases from X-ray images, generates reports, and provides visual explanations using Grad-CAM.

Built a production-ready AI system integrating model training, explainability, and deployment.

## 🔗 Live Demo

👉 https://jaimin17-chest-xray-ai.hf.space/

Upload a chest X-ray image to see predictions, Grad-CAM, and report.

---

## 🚀 Features

- Multi-label disease prediction (5 conditions)
- Grad-CAM visualization (model explainability)
- Automated report generation
- Interactive web UI (Streamlit)
- Backend API (FastAPI)
- Dockerized and deployed (Hugging Face Spaces)

---

## 🧠 Model

- Architecture: DenseNet121 (pretrained)
- Input: 224 × 224 chest X-ray images
- Loss: BCEWithLogitsLoss (multi-label)
- Labels:
  - Atelectasis
  - Cardiomegaly
  - Consolidation
  - Edema
  - Pleural Effusion

---

## 🏗️ Tech Stack

- PyTorch, NumPy, Pandas
- FastAPI (Backend)
- Streamlit (Frontend)
- Grad-CAM (Explainability)
- Docker (Deployment)

---

## ⚙️ Run Locally

### Backend

    uvicorn app.backend.main:app --reload

### Frontend

    streamlit run app/frontend/streamlit_app.py

---

## 🐳 Run with Docker

    docker build -t chest-xray-ai .
    docker run -p 8000:8000 -p 8501:8501 chest-xray-ai

---

## 📌 Note

- This project uses a subset of the CheXpert dataset for training.
- Designed for **academic and research purposes**.

---

## ⚠️ Disclaimer

This tool is **not a medical diagnosis system** and should not be used for clinical decisions.

---

## 👤 Author

Jaimin Patel  
MS Data Science — DePaul University