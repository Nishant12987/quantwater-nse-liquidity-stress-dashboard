# **Execution Guide: Harsh Gautam (Validation & Microstructure)**

**Role:** The "Validation" Pillar  
**Focus:** Microstructure Analysis, Model Robustness, EDA, and Dashboard UI/UX.

### **Strategic Mission**
Your goal is to ensure the mathematical truth of the liquidity metrics and the system's robustness under market stress. As a former Test Engineer, you are the "quality gatekeeper." You will utilize a **Hybrid-Lean workflow**, performing rigorous data validation and UI development locally, while offloading heavy Explainable AI (SHAP) tasks to the cloud to save **3GB+ of disk space** and utilize GPU acceleration.

---

### **Weekly Execution Breakdown**

*   **Week 1: Data Integrity & PIT Audit — [Local/Lean]**
    *   **Audit Bhavcopy Data:** Review raw files from Nishant’s scraper. Verify "Point-in-Time" integrity.
    *   **Ticker Mapping:** Handle ticker changes and corporate actions locally. 
    *   **Persistence:** Ensure the "Validated_Raw_Data" is synced to the **Shared Google Drive** so Nishant can begin cloud training.

*   **Week 2: Microstructure Logic Verification — [Local/Lean]**
    *   **Formula Verification:** Implement math for **Amihud Illiquidity**, **Roll Spread**, and **Price Impact** using the lean stats stack.
    *   **Academic Benchmarking:** Compare calculated metrics against published research to ensure 100% calculation accuracy.
    *   **Handoff:** Provide the validated "Feature Calculation Logic" to Nishant to build the scaled Feature Factory.

*   **Week 3: Stress Regime Labeling (EDA) — [Local/Lean]**
    *   **Thresholding:** Define "Liquidity Stress" mathematically (e.g., 2.5σ deviations from 60-day MA).
    *   **Validation:** Perform Exploratory Data Analysis (EDA) locally to confirm these labels align with high-volatility days (VIX spikes) using the processed datasets from Drive.

*   **Week 4: Model Performance Audit — [Local/Lean]**
    *   **Output Audit:** Download Nishant’s cloud-generated model outputs from Drive.
    *   **Sectoral Bias Check:** Evaluate XGBoost precision and LSTM RMSE locally across different NSE sectors. Adjust thresholds if necessary.

*   **Week 5: EXAM BREAK (No Work)**

*   **Week 6: Historical Shock Backtesting — [Local/Lean]**
    *   **Deep-Dive Analysis:** Run the ensemble engine locally (using weights from Drive) through the 2020 COVID-19 crash and 2024 volatility spikes.
    *   **Lead-Lag Analysis:** Determine if the system provides a "Lead Indicator" for spread widening.

*   **Week 7: UI/UX Dashboard Design — [Local/Lean]**
    *   **Frontend Build:** Design the visual layout in Streamlit locally. 
    *   **Traders' View:** Build the "Liquidity Health Gauges" (Green/Amber/Red) and interactive sector heatmaps.

*   **Week 8: Cloud XAI (SHAP) Analysis — [Cloud/GPU]**
    *   **Cloud Inference:** Open the Colab notebook and install `requirements-colab.txt`.
    *   **Heavy Extraction:** Run the `shap` library in the cloud (T4 GPU) to identify primary drivers for specific alerts. This offloads the massive memory requirement from your laptop.
    *   **Visual Generation:** Download SHAP results and generate high-res Precision-Recall curves locally for the paper.

*   **Week 9: Academic Writing & Finalization — [Local/Lean]**
    *   **Drafting Theory & Results:** Lead the writing for "Microstructure Theory" and "Empirical Results" in LaTeX.
    *   **Final QA:** Perform a final end-to-end test of the Streamlit app before deployment.

*   **Week 10: EXAM BREAK (No Work)**

---

**Operational Syncs:**
*   **Infrastructure:** Do not install `tensorflow` or `shap` locally. Use your local environment for `pandas`, `scipy`, and `streamlit` only.
*   **With Nishant:** Receive raw data in **Week 1**. Deliver validated "Logic" in **Week 2**. Access his cloud-trained model weights on **Drive** starting **Week 4**.
*   **With Pawan:** Sync in **Week 3** to ensure "Liquidity Commonality" aligns with Prof. Tripathi's research requirements.

**Tooling:** `Google Colab` (XAI), `Google Drive` (Sync), `Streamlit` (Dashboard), `SciPy` (Math).