# **Execution Guide: Nishant Ameta (Architecture & Deployment)**

**Role:** The "Machine" Pillar  
**Focus:** ML Engineering, Data Ingestion, Pipeline Automation, and Dashboard Deployment.

### **Strategic Mission**
Your goal is to build the "Engine Room" of the project. You are responsible for the structural integrity, automation, and cloud-hybrid deployment of the system. Leveraging your experience in operational analytics, you will implement a **Hybrid-Lean workflow**: performing data ingestion and pipeline scripting locally while offloading heavy ML training (XGBoost GPU/LSTM) to the cloud. This strategy ensures high-performance results while saving **3GB+ of local disk space**.

---

### **Weekly Execution Breakdown**

#### **Week 1: Bhavcopy Scraper Automation — [Local/Lean]**
*   **Infrastructure Setup:** Setup the shared team Google Drive folder. Configure a `config.py` script to handle dynamic path switching (`/content/drive/...` for Colab vs. `./data/...` for Local).
*   **Scraper Development:** Script a robust scraper to download historical NSE Bhavcopy (Equity) zip files from 2021–2026. 
*   **Pipeline Setup:** Automate the extraction of CSVs from zips and initial cleaning (removing 'Series', 'ISIN').
*   **Persistence:** Ensure the raw data is synced to **Google Drive** for team access.

#### **Week 2: High-Scale Feature Factory — [Local/Lean]**
*   **Scaling:** Take the validated mathematical logic from Harsh and implement it using vectorized `Pandas` or `NumPy` operations locally.
*   **Market Context:** Calculate market-wide liquidity averages to enable the "Liquidity Commonality" feature.
*   **Feature Matrix:** Upload the finalized, engineered dataset (top 100 stocks) to **Drive** to prepare for cloud modeling.

#### **Week 3: Cloud-Based XGBoost Stress Gatekeeper — [Cloud/GPU]**
*   **Colab Training:** Open a Google Colab notebook and install `8-requirements-colab.txt`.
*   **Binary Training:** Train the XGBoost classifier using the `gpu_hist` method for acceleration.
*   **Optimization:** Use GridSearchCV to optimize for **Precision**. 
*   **Persistence:** Save the trained model weights (`.pkl`) back to the **Drive `/models/checkpoints/`** folder.

#### **Week 4: Cloud-Based LSTM Sequence Forecaster — [Cloud/GPU]**
*   **Space Saving:** Build and train the LSTM architecture in **Google Colab** (utilizing the Cloud TensorFlow stack to avoid a 2GB+ local installation).
*   **Tensor Engineering:** Reshape Drive-stored CSVs into 3D Tensors using a 10-day rolling lookback.
*   **Persistence:** Serialize the trained LSTM model (`.h5` or `.json`) directly to **Google Drive**.

#### **Week 5: EXAM BREAK (No Project Work)**

#### **Week 6: Ensemble Integration Logic — [Local/Lean]**
*   **Model Loading:** Develop the `EnsembleManager` class locally, configured to load pre-trained weights from the Drive bridge.
*   **Alert Logic:** Combine XGBoost classification with LSTM forecasting. If classification is "Stress" AND predicted spread > threshold, trigger "Critical Alert."
*   **Inference Pipeline:** Optimize the pipeline to process a new day's Bhavcopy in under 5 seconds.

#### **Week 7: Dashboard Backend & Model API — [Local/Lean]**
*   **Streamlit Backend:** Build the backend logic in Streamlit. Connect the local inference engine to the model weights on Drive.
*   **Caching:** Implement `st.cache_data` to ensure the dashboard loads historical data and heatmaps instantly.
*   **Deliverable:** Functional local backend ready for Harsh’s UI components.

#### **Week 8: Deployment & CI/CD — [Local/Lean]**
*   **GitHub Actions:** Set up a `.github/workflows` file to automate ingestion testing.
*   **Public Repository:** Organize the repo with a professional structure. Finalize documentation for institutional visibility.
*   **Streamlit Cloud:** Deploy the final app to Streamlit Cloud, ensuring Drive connectivity is secure.

#### **Week 9: Technical Specs & Architecture Writing — [Local/Lean]**
*   **Drafting:** Lead the writing of "ML Architecture," "Ensemble Methodology," and "Cloud Infrastructure" for the papers.
*   **Final Review:** Ensure the GitHub `requirements.txt` and `README.md` are perfectly aligned with the deployed product.

#### **Week 10: EXAM BREAK (No Project Work)**

---

### **⚡ Hybrid-Lean Operational Syncs:**
1.  **Disk Space Management:** **Do not install** `tensorflow` or `shap` locally. Perform all heavy training and XAI passes in Google Colab.
2.  **Infrastructure Lead:** You own the `/models/` and `/data/` directory structure on Google Drive. Ensure Harsh can load your trained weights into his local Streamlit UI seamlessly.
3.  **With Harsh:** Provide him with the raw dataset in **Week 1**. Retrieve his validated "Logic" in **Week 2** to build the Feature Factory.
4.  **With Pawan:** Sync in **Week 8** to finalize the public GitHub repository for institutional visibility.

**Tooling:** `Google Colab` (GPU Training), `Google Drive` (Sync), `FastAPI` (Local), `Streamlit` (Dashboard), `GitHub Actions` (CI/CD).