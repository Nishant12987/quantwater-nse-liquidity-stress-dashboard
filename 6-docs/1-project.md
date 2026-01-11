# **Project: ML Liquidity Prediction Dashboard (NSE)**
### *The Microstructure Frontier: Ensemble Learning for Market Stability*

---

## **1. The Mission**
In equity markets, liquidity is the **"oxygen"** that allows trading to function. When liquidity evaporates, spreads widen, price impact spikes, and systemic risk increases. For a trader or an institutional investor, entering a position is easy, but exiting during a "Liquidity Drought" is where fortunes are lost.

Traditional liquidity monitoring is reactive—it tells you when the market *is* stressed. Our project, the **ML Liquidity Prediction Dashboard**, aims to be proactive. We believe that by analyzing high-frequency microstructure signals, we can predict "liquidity choke points" before they manifest. Our mission is to build a **Production-Grade Monitoring System** for the National Stock Exchange (NSE) that achieves **80%+ precision** in identifying stress events.

---

## **2. The Methodology: The Ensemble Advantage**
We are moving beyond simple volume charts to a sophisticated **Ensemble ML Architecture** that utilizes cloud-accelerated computing:

*   **XGBoost (The Gatekeeper):** A high-precision binary classifier trained to distinguish between "Normal" and "Stress" regimes. It acts as the primary alert system, catching 80%+ of historical stress events like the COVID-19 crash and the 2024 volatility spikes.
*   **LSTM (The Forecaster):** A Deep Learning sequence model trained in **Google Colab** to capture the "temporal memory" of the market. It predicts the **next-day bid-ask spread** with a target RMSE of <5%, validating the XGBoost alerts with continuous forecasting.
*   **Microstructure Features:** We are engineering "Deep Features" including the **Amihud Illiquidity Ratio**, **Roll Spread**, **Price Impact**, and **Liquidity Commonality**, calculated locally for maximum efficiency.

---

## **3. Focus: The National Stock Exchange (NSE)**
Our primary laboratory is the Indian equity market. We are leveraging five years of **NSE Bhavcopy data (2021-2026)** to train our models. This project is specifically designed to handle the unique "High-Growth, High-Volatility" characteristics of the NSE, ensuring our alerts are relevant for Nifty 50 and Next 50 stocks.

---

## **4. The Technical Pillars (Hybrid-Lean Roles)**

To succeed, we use a **Hybrid-Lean** infrastructure. We save **3GB+ of local disk space** by offloading heavy libraries (TensorFlow/SHAP) to the cloud. Your roles are adapted to this workflow:

#### **Pillar 1: The Architecture & Deployment (Nishant Ameta)**
*   **Goal:** Build the engine and the interface.
*   **Cloud Perspective:** You are the **Infrastructure Architect**. You automate the NSE Bhavcopy ingestion locally and manage the **Google Drive Data Bridge**. You design and train the **XGBoost/LSTM** ensemble in **Google Colab** using T4 GPUs, ensuring model weights are saved back to Drive for the dashboard.

#### **Pillar 2: The Validation & Microstructure (Harsh Gautam)**
*   **Goal:** Ensure mathematical truth and system robustness.
*   **Perspective:** You are the **Quality Gatekeeper**. You ensure our "Feature Factory" (Amihud/Roll spread) is academically sound using local math tools. You use **Cloud-Based XAI (SHAP)** to prove our models aren't "black boxes," then download the results to build an intuitive **Streamlit Dashboard** locally.

---

## **5. The Deliverables (The "Big Wins")**
By the end of Week 9, we will have achieved three massive milestones:
1.  **Two Academic Papers:** Two working papers suitable for arXiv/conference submission, focusing on *Ensemble Learning in Microstructure* and *Predictive Liquidity Commonality*.
2.  **Live Streamlit Dashboard:** A publicly accessible production system providing real-time liquidity health scores for NSE stocks.
3.  **A Professional Portfolio:** A public GitHub repository demonstrating technical depth in ML Ensembles, Microstructure, and Cloud Deployment.

---

## **6. A Note on the Workflow & Timeline**
This is a light-mid-intensity 10-week sprint supported by **Cloud Infrastructure**.

**The Hybrid-Lean Workflow:**
1.  **Code Locally:** Write scraping and microstructure scripts in your lean local environment (no TensorFlow).
2.  **Train in Cloud:** Open Colab notebooks to run heavy LSTM training or SHAP analysis using free GPU credits.
3.  **Sync via Drive:** Use the shared Google Drive folder as the central "Hard Drive" for raw data and model checkpoints.

**Remember:** 
*   **Nishant’s** data bridge feeds the **Cloud Ensemble**.
*   **Harsh’s** validated features improve the **Cloud Accuracy**.
*   **Both** collaborate locally to turn these results into a live, deployed product.

**Welcome to the Liquidity Dashboard team. Let’s build the future of market microstructure monitoring.**

## 📂 Next Step: Project Overview
See [2.overview.md](2.overview.md) for the full project roadmap and compute split.