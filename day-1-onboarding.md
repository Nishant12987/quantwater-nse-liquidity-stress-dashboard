# 🚀 Day 1: Project Onboarding & Infrastructure Setup

**Welcome Team!** We are moving into a high-performance phase using a **Hybrid-Lean Workflow**. This strategy allows us to use world-class AI (Transformers, LSTMs, FinBERT) while saving **3-4GB of disk space** on your local machines and utilizing T4 GPUs in the cloud.

---

### 📂 Phase 1: Storage Bridge (Shared Google Drive)
Our "Hard Drive" for this project is not your laptop; it is a shared Google Drive folder. This ensures that a model trained by one person in the Cloud can be analyzed by another locally.

1.  **Action for Machine Leads (Prolay & Nishant):** 
    *   Create a folder named `quantwater_projects_2026`.
    *   Create subfolders: `/project_1_rotation/` and `/project_2_liquidity/`.
    *   Share these folders with the entire team (including Pawan) with **Editor** access.
2.  **Action for All Interns:**
    *   "Add a Shortcut" to these folders in your own Google Drive.
    *   Install **Google Drive for Desktop**. This allows you to access files at `G:/My Drive/...` just like a local folder.

---

### 💻 Phase 2: GitHub Repository Setup
1.  **Clone the Repo:** 
    *   **Project 1 Team:** Clone `global-dynamic-factor-rotation-v4`
    *   **Project 2 Team:** Clone `nse-liquidity-stress-dashboard`
2.  **Local Folder Structure:** Ensure your local directory matches the repo structure exactly. 
3.  **Path Management:** Open `src/config.py`. Ensure the `ROOT_PATH` correctly points to your synced Google Drive folder.

---

### 🛠 Phase 3: Environment Installation (The Lean Stack)
**Do not install heavy libraries locally.** Follow these steps to save space:

**For Project 1 (Amogh, Jasmair, Prolay):**
```bash
# Create local environment
python -m venv v4_env
source v4_env/bin/activate # (or .\v4_env\Scripts\activate on Windows)

# Install LEAN requirements (No Torch/Transformers)
pip install -r requirements.txt
```

**For Project 2 (Harsh, Nishant):**
```bash
# Create local environment
python -m venv nse_env
source nse_env/bin/activate

# Install LEAN requirements (No TensorFlow/SHAP)
pip install -r 8-requirements.txt
```

---

### ☁️ Phase 4: Cloud Validation (Google Colab)
Verify that the heavy-lifting pipeline is ready:

1.  Open [colab.research.google.com](https://colab.research.google.com).
2.  Create a new notebook and set **Runtime Type -> T4 GPU**.
3.  **Run the Setup Cell:**
    *   **P1:** `!pip install -r requirements-colab.txt`
    *   **P2:** `!pip install -r 8-requirements-colab.txt`
4.  **Mount Drive:** Run `from google.colab import drive; drive.mount('/content/drive')`.
5.  If you can see the shared project folder, the bridge is working.

---

### 🎯 Day 1 Deliverables (EOD Check)

**Team 1 (Global Factor Rotation):**
*   **Prolay:** Establish the Drive directory and sync the `v3-archive` macro data.
*   **Jasmair:** Successfully mount Drive in Colab and run a "Hello World" GPT-4 API call.
*   **Amogh:** Audit the PIT timestamps in the `v3-archive` files.

**Team 2 (Liquidity Dashboard):**
*   **Nishant:** Successfully download the first 1 month of NSE Bhavcopy zip files to the shared Drive.
*   **Harsh:** Create a local Jupyter notebook that reads one Bhavcopy file from Drive and plots a basic Volume Distribution.

---

### 💡 The Golden Rule:
**If it's heavy (Deep Learning/NLP), do it in Colab. If it's logic (Scraping/Math/UI), do it locally.** 

Always `git push` your code changes and `drive sync` your data changes. 

**Let’s build!**