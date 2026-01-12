# **Tasks Guide: Harsh Gautam**
**Role:** Validation & Microstructure Specialist (Validation Pillar)  
**Focus:** Microstructure Analysis, Model Robustness, EDA, and Dashboard UI/UX (Hybrid-Lean Workflow).

## 📂 Full individual execution:
*to execute the tasks mentioned below check:* [2-individual-execution/harsh-execution.md](../2-individual-execution/harsh-execution.md)

## Tasks:
| Week | Task Description | Deliverable |
| :--- | :--- | :--- |
| **W1** | **Data Integrity & PIT Audit (Local/Lean):** Audit raw NSE Bhavcopy data for "Point-in-Time" integrity. Handle ticker mapping and corporate actions (splits/bonus) locally to ensure the backtest is free of data leakage. | Cleaned Master Stock Database. |
| **W2** | **Microstructure Logic Verification (Local/Lean):** Define the mathematical logic for **Amihud Illiquidity**, **Roll Spread**, and **Price Impact**. Verify local calculations against academic benchmarks using the lightweight stats stack. | Validated Feature Calculation Logic. |
| **W3** | **Stress Regime Labeling (Local/Lean):** Define statistical thresholds for "Liquidity Stress." Perform local EDA to validate that labels correlate with high-volatility days (VIX spikes). | Stress Labeling Logic & EDA Report. |
| **W4** | **Model Performance Audit (Local/Lean):** Audit Nishant’s cloud-generated model outputs. Evaluate the XGBoost precision and LSTM RMSE locally. Check for consistency across NSE sectors. | Model Performance Audit. |
| **W5** | **EXAM BREAK** | **No Work** |
| **W6** | **Historical Shock Backtesting (Local/Lean):** Conduct a deep-dive backtest of the ensemble engine against historical events (COVID, Budget Days) using the local backtest engine. | Historical Stress Validation Report. |
| **W7** | **Dashboard UI/UX Design (Local/Lean):** Build the Streamlit interface locally. Develop "Liquidity Health Gauges," sector heatmaps, and search functionality for Nifty 50 tickers. | Functional Dashboard UI Prototype. |
| **W8** | **Cloud Explainability Analysis (Cloud/GPU):** Use **Google Colab** to run heavy **SHAP value** extractions to identify alert drivers. Download results to generate high-res visuals locally. | Visual Asset Pack (Cloud-driven). |
| **W9** | **Academic Writing (Local/Lean):** Lead the "Microstructure Theory," "Data Analysis," and "Empirical Results" sections of the two research papers. Perform final QA on the Live App. | Final Draft (Theory & Results). |
| **W10** | **EXAM BREAK** | **No Work** |

---

### **⚡ Hybrid-Lean Operational Notes for Harsh:**
1. **Local Space Management:** You only need to install the lightweight stack (`8-requirements.txt`). Do not install TensorFlow or SHAP locally to save ~3GB of disk space.
2. **Mathematical Validation:** Perform all microstructure formula testing locally using `numpy` and `scipy`. These are lightweight and highly efficient for CPU-based validation.
3. **The Cloud Bridge:** Conduct the final explainability pass (Week 8) on **Google Colab** to utilize the GPU for SHAP's intensive KernelExplainer tasks.
4. **Data Retrieval:** Access raw data and Nishant's cloud-trained model weights via the **Shared Google Drive** to ensure your local dashboard uses the most accurate "Brain."