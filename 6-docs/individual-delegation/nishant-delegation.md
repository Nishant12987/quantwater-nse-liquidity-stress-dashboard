# **Tasks Guide: Nishant Ameta**
**Role:** Architecture & Deployment Specialist (Machine Pillar)  
**Focus:** ML Engineering, Data Ingestion, and Dashboard Deployment (Hybrid-Lean Workflow).

| Week | Task Description | Deliverable |
| :--- | :--- | :--- |
| **W1** | **Bhavcopy Scraper Automation (Local/Lean):** Develop the automated scraper for NSE Bhavcopy (2021–2026). Setup the **Google Drive Bridge** for persistent data storage and team synchronization. | Automated Data Pipeline + Drive Sync. |
| **W2** | **High-Scale Feature Factory (Local/Lean):** Script the vectorized "Feature Factory" module to generate indicators (Amihud, Roll, Commonality) at scale. Upload finalized feature-engineered matrices to Drive. | Feature Engineering Engine. |
| **W3** | **Cloud-Based XGBoost Implementation (Cloud/GPU):** Train the XGBoost binary classifier in **Google Colab** using `gpu_hist`. Optimize hyperparameters for **Precision** to minimize false-positive alerts. | XGBoost Stress Gatekeeper (on Drive). |
| **W4** | **Cloud-Based LSTM Construction (Cloud/GPU):** Build and train the LSTM architecture in **Google Colab** (utilizing the Cloud TensorFlow stack to save 3GB+ local space). Implement 3D windowing. | Predictive LSTM Engine (on Drive). |
| **W5** | **EXAM BREAK** | **No Work** |
| **W6** | **Ensemble Integration (Local/Lean):** Develop the `EnsembleManager` logic locally, loading weights from Drive. Combine classification regimes with forecast dynamics into a single health score. | Ensemble Alert Logic. |
| **W7** | **Dashboard Backend & API (Local/Lean):** Build the Streamlit backend logic. Connect the local inference engine to the model checkpoints stored on Google Drive for live inferencing. | Functional Dashboard Backend. |
| **W8** | **Deployment & CI/CD (Local/Lean):** Finalize the GitHub repo structure. Setup **GitHub Actions** to automate ingestion script testing and Streamlit Cloud deployment. | CI/CD Pipeline & GitHub Repo. |
| **W9** | **Technical Writing (Local/Lean):** Lead the "ML Architecture," "Ensemble Methodology," and "Deployment Strategy" sections of the research papers. | Final Draft (Architecture & Deployment). |
| **W10** | **EXAM BREAK** | **No Work** |

---

### **⚡ Hybrid-Lean Operational Notes for Nishant:**
1. **Local Space Management:** You are the infrastructure lead. **Do not install** `tensorflow` or `shap` locally. This saves ~3GB of space and ensures you build models that are portable via the cloud.
2. **Persistence Strategy:** Every training script in Colab must end with a cell that saves the model weights (`.pkl`, `.h5`, or `.json`) directly to the **Shared Google Drive** folder.
3. **Data Bridge Ownership:** You are responsible for ensuring the `/data/` directory structure on Drive mirrors the local environment, allowing `config.py` to switch paths seamlessly.
4. **Environment Consistency:** Maintain the `8-requirements.txt` (Local) and `8-requirements-colab.txt` (Cloud) files so Harsh and the supervisor can replicate your results without version conflicts.