# **Roadmap.md**
*High-level strategic roadmap for the NSE Liquidity Stress Prediction Dashboard (Hybrid-Lean Workflow).*

---

### **⚡ Compute Infrastructure Overview**
To maximize performance and save **3GB+ of local disk space**, this roadmap utilizes a **Hybrid-Lean** approach:
*   **Local (Lean):** Data scraping, cleaning, microstructure math, and Streamlit UI development.
*   **Cloud (Colab T4 GPU):** Heavy TensorFlow training for LSTM, GPU-accelerated XGBoost, and SHAP explainability.
*   **Storage (Google Drive):** Persistent "Digital Hard Drive" for NSE Bhavcopy datasets, feature matrices, and model checkpoints.

---

#### **Part 1: Data Ingestion & Microstructure Factory (Local/Lean)**
*   **Week 1: Bhavcopy Automation & Pipeline Setup**
    *   **Data Sourcing:** Script automated local downloads of NSE Bhavcopy (2021–2026).
    *   **Cleaning:** Handle ticker changes and corporate actions locally. Sync cleaned data to Google Drive.
    *   **Deliverable:** Automated raw data pipeline and master database synced to Drive.
*   **Week 2: Advanced Feature Engineering**
    *   **Vectorized Math:** Implement Amihud Illiquidity, Roll Spread, and Price Impact calculations using local CPU resources.
    *   **Market Context:** Compute Liquidity Commonality metrics. Upload the finalized feature-engineered matrix to Drive.
    *   **Deliverable:** Master dataset with 20+ indicators ready for cloud modeling.

#### **Part 2: Ensemble Model Development (Cloud/GPU)**
*   **Week 3: Binary Classification - XGBoost (Cloud/GPU)**
    *   **Colab Training:** Open a Colab notebook, mount Drive, and train the XGBoost classifier using the `gpu_hist` tree method for speed.
    *   **Target:** Achieve 80%+ precision in identifying "Stress" regimes.
    *   **Deliverable:** Trained classifier weights serialized to `models/checkpoints/` on Drive.
*   **Week 4: Time-Series Forecasting - LSTM (Cloud/GPU)**
    *   **DL Construction:** Use **TensorFlow/Keras** in Colab (to avoid local installation) to build the LSTM spread engine.
    *   **Validation:** Use walk-forward validation in the cloud to minimize forecast RMSE.
    *   **Deliverable:** Predictive LSTM engine weights saved to Drive.

#### **Week 5: EXAM BREAK (No Work)**

#### **Part 3: System Integration & Dashboarding (Hybrid)**
*   **Week 6: Ensemble Alert Logic (Local/Lean)**
    *   **Integration:** Develop the ensemble logic locally. Test the "Red Alert" trigger using model weights downloaded from Drive.
    *   **Backtest:** Validate against historical shocks (COVID, Budget Days) using the local backtest engine.
    *   **Deliverable:** Robust alert engine logic and backtest result logs.
*   **Week 7: Streamlit UI Development (Local/Lean)**
    *   **Product Build:** Build the Streamlit dashboard locally using the lean stack.
    *   **Features:** Implement sectoral heatmaps and health gauges. Connect to the local inference API.
    *   **Deliverable:** Functional beta version of the Live Dashboard.
*   **Week 8: Performance Metrics & Cloud XAI (Hybrid)**
    *   **Explainability (Cloud):** Run heavy **SHAP** passes in Colab to identify liquidity stress drivers.
    *   **Visuals (Local):** Use SHAP results to generate publication-quality charts and app-ready visuals locally.
    *   **Deliverable:** Finalized visualization suite for the papers and dashboard.

#### **Part 4: Writing & Public Launch (Local/Lean)**
*   **Week 9: Double-Paper Drafting & Deployment**
    *   **Paper 1 & 2:** Draft "Ensemble ML Architecture" and "Predictive Commonality" papers using LaTeX.
    *   **Deployment:** Deploy the dashboard to Streamlit Cloud. Finalize the public GitHub repository documentation.
    *   **Deliverable:** 2 arXiv-ready papers + Live Dashboard URL.

#### **Week 10: EXAM BREAK (No Work)**

## 📂 Next Step: Project Ownership
See [3-ownership.md](3-ownership.md) for the Harsh & Nishant task breakdown and infrastructure syncs.