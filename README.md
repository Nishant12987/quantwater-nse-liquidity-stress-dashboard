# NSE Liquidity Stress Prediction Dashboard (V1)
### Ensemble ML Strategy (XGBoost + LSTM) for Microstructure Monitoring

**Quantwater Tech Investments | IIT Kanpur Capstone Project**

## 🚀 Mission
Building a production-grade monitoring system for the National Stock Exchange (NSE) to predict liquidity "choke points" with **80%+ precision**. We utilize an ensemble of **XGBoost and LSTM** to identify microstructure decay and provide real-time stress alerts for Indian equity markets.

## ⚡ Compute Infrastructure (Hybrid-Lean)
To optimize performance and save **3GB+ of local disk space**, this project utilizes a split-environment strategy:

1.  **Local Environment (Lean):** Laptops run a lightweight stack for NSE data scraping, microstructure feature engineering (Amihud/Roll), and Streamlit UI development.
2.  **Cloud Environment (High-Performance):** **Google Colab (T4 GPUs)** is used for training the XGBoost classifier and the LSTM time-series forecaster.
3.  **Data Bridge:** **Google Drive** serves as the persistent storage layer for NSE Bhavcopy files, processed indicators, and trained model weights.

## 🛠 Tech Stack
- **Languages:** Python (Pandas, NumPy, Scikit-Learn, Statsmodels)
- **ML Models:** XGBoost (Classification), TensorFlow (LSTM Time-Series) [Cloud Only]
- **Data Engine:** NSE Bhavcopy (2021-2026), Amihud/Roll/Commonality Metrics
- **Deployment:** Streamlit (Live Dashboard), GitHub Actions (CI/CD)
- **Research:** LaTeX (Academic Paper Drafting)

## 👥 The Team
- **Pawan (Lead):** Project Oversight & Research Supervision.
- **Harsh Gautam (Validation):** Microstructure Specialist & Model Validation (UX/UI Design).
- **Nishant Ameta (Architecture):** ML Engineering & Pipeline Automation (Cloud Infrastructure).

## 📅 Roadmap (8-Week Sprint)
- **Weeks 1-2:** Bhavcopy Scraper Automation & **Feature Factory (Amihud/Roll/Commonality)**.
- **Weeks 3-4:** **Cloud-Based Ensemble Training (Colab/GPU)**: XGBoost + LSTM.
- **Week 5:** *Exam Break*
- **Weeks 6-7:** Multi-Stock Backtesting & Historical Stress Event Validation (2024 Volatility).
- **Week 8:** **Cloud XAI (SHAP) Analysis** & Streamlit UI Real-Time Deployment.
- **Week 9:** arXiv Submission (Two 6-7 page papers) & Final Product Launch.
- **Week 10:** *Exam Break*

## 💻 Installation & Setup

### 1. Local Environment (Laptop)
Install the lean stack for data cleaning and dashboard building without the 3GB overhead of TensorFlow:
```bash
pip install -r 8-requirements.txt
```

### 2. Cloud Environment (Google Colab)
Run this command in your Colab Notebook to enable GPU-accelerated modeling:
```python
!pip install -r 8-requirements-colab.txt
```

## 📂 Project Narrative
See [docs/1-project.md](docs/1-project.md) for the full project vision.