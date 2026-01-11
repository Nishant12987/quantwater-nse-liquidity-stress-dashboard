# **3-delegation.md**
*Co-lead task allocation and responsibilities for the NSE Liquidity Dashboard (Hybrid-Lean Workflow).*

## 📂 Full individual delegation:   
* **Harsh:** See [individual-delegation/harsh-delegation.md](individual-delegation/harsh-delegation.md)
* **Nishant:** See [individual-delegation/nishant-delegation.md](individual-delegation/nishant-delegation.md)

## 📂 Full individual execution:
*to execute the tasks mentioned in the delegation files.*
* **Harsh:** See [individual-execution/harsh-execution.md](individual-execution/harsh-execution.md)
* **Nishant:** See [individual-execution/nishant-execution.md](individual-execution/nishant-execution.md)

## 📂 Overall delegation:   

#### **Week 1: Bhavcopy Automation & Pipeline Setup — [Local/Lean]**
*   **Nishant:** Develop the automated scraper for NSE Bhavcopy (2021–2026). Set up the **Google Drive Bridge** for persistent data storage and team sync.
*   **Harsh:** Audit the raw data for "Point-in-Time" integrity. Handle ticker mapping and corporate actions locally. Ensure all raw CSVs are correctly structured on Drive.

#### **Week 2: Advanced Feature Engineering (The Factory) — [Local/Lean]**
*   **Harsh:** Define the mathematical implementation of **Amihud Illiquidity**, **Roll Spread**, and **Liquidity Commonality**. Verify calculations using local CPU resources.
*   **Nishant:** Script the vectorized "Feature Factory" module to generate indicators for the top 100 NSE stocks. Upload finalized feature-engineered matrices to Drive.

#### **Week 3: Binary Classification (XGBoost Gatekeeper) — [Cloud/GPU]**
*   **Nishant:** Train the XGBoost binary classifier in **Google Colab** using `gpu_hist` for acceleration. Optimize for precision to minimize false alarms.
*   **Harsh:** Analyze the classification results locally. Validate that predicted stress regimes align with high-volatility days (VIX spikes). Save weights to Drive.

#### **Week 4: Time-Series Forecasting (LSTM Spread Engine) — [Cloud/GPU]**
*   **Nishant:** Build and train the LSTM architecture in **Google Colab** (utilizing the Cloud TensorFlow stack to save 3GB+ local space). 
*   **Harsh:** Evaluate forecasting accuracy (RMSE/MAE) locally. Perform sectoral audits to ensure no bias in low-volume stock predictions.

#### **Week 5: EXAM BREAK**

#### **Week 6: Ensemble Alert Logic & Historical Backtest — [Local/Lean]**
*   **Nishant:** Develop the `EnsembleManager` logic locally, loading weights from Drive. Combine classification regimes with forecast dynamics.
*   **Harsh:** Conduct the historical deep-dive backtest locally. Validate system response to the COVID-19 shock and 2024-25 Budget Day volatility.

#### **Week 7: Streamlit UI Development — [Local/Lean]**
*   **Nishant:** Build the Streamlit backend logic. Connect the local inference engine to the model checkpoints stored on Google Drive.
*   **Harsh:** Design the dashboard UI/UX locally. Build the "Liquidity Health Gauges" and sector-wise heatmaps for real-time monitoring.

#### **Week 8: Explainability (XAI) & Performance Visuals — [Hybrid]**
*   **Harsh:** Execute heavy **SHAP value** computations in **Google Colab** to explain alert drivers. Download results to generate visuals locally.
*   **Nishant:** Finalize the GitHub repository structure and automate the deployment pipeline via GitHub Actions.

#### **Week 9: Paper Writing & Public Launch — [Local/Lean]**
*   **Writing Roles:** **Harsh** (Microstructure Theory/Analysis), **Nishant** (ML Architecture/Deployment).
*   **Joint:** Submit two arXiv-ready papers and launch the Live Production Dashboard on Streamlit Cloud.

#### **Week 10: EXAM BREAK**

---

### **Critical Success Factors**
1.  **Hybrid Efficiency:** Saving 3GB+ disk space per co-lead by offloading TensorFlow and SHAP to Google Colab.
2.  **Ensemble Precision:** Achieving 80%+ accuracy in "Red Alerts" by combining XGBoost classification with LSTM temporal forecasting.
3.  **Drive-Sync Consistency:** Maintaining a seamless data bridge between local scraping/feature engineering and cloud model training.
4.  **Actionable Microstructure:** Providing a "Liquidity Health Score" that translates complex Amihud and Roll metrics into intuitive gauges for traders.