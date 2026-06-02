## 🎓 Academic Context
This module represents the core Machine Learning & Inference component developed for my **Bachelor's Degree (Licence en Informatique)** graduation project (PFE).

* **Institution:** University of Science and Technology Houari Boumediene (USTHB), Algiers.
* **Degree:** Licence en Informatique (Academic Bachelor's Degree in Computer Science)
* **Academic Year:** 2025-2026
* **Project Topic:** Development of an Intelligent Supermarket with a Connected Shopping Cart and an Intelligent Recommendation System.
* **Supervised by:** Pr. BOUYAKOUB Fayçal M'hamed


# Smart Supermarket - Hybrid Recommendation Engine (ML Microservice)

This repository contains the standalone, high-performance Machine Learning recommendation core developed for the **Smart Supermarket** retail ecosystem. Built entirely as a decoupled ML microservice using **Flask**, it delivers real-time, context-aware, anti-waste, and personalized product recommendations to intelligent frontend clients and IoT connected shopping carts.

---

## 👥 Hybrid Recommender System Profiles (The 3 Multi-Modal Types)

To maximize accuracy and natively bypass the classic *Cold Start* problem, the engine dynamically adapts its hybrid scoring and ranking pipelines into **3 operational profiles** based on active session inputs and authentication states:

### 1️⃣ Type 1: Anonymous / Guest Profile (100% Context-Aware)
* **Scenario:** Triggered for guest sessions, unauthenticated shoppers, or new lookups where historical interaction data is completely absent.
* **Logic:** The engine shifts 100% of the scoring weight to external environmental and contextual vectors:
  * **Temporal Sync ($H_i$):** Matches popularity metrics specific to the active hour of the day.
  * **Seasonal Sync ($S_i$):** Ingests current macro seasonal trends.
  * **Anti-Waste Boost ($D_i$):** Prioritizes items near their expiration date (DLC) to accelerate clearance.

### 2️⃣ Type 2: New Authenticated Users (Session-Based + Context with Pre-Filtering)
* **Scenario:** Activated when a user logs in but has zero long-term historical purchase or interaction records.
* **Filtering & Logic (Pre-Filtering):** * **How it works:** The multi-layer health, diet, and allergen guardrails are executed **BEFORE** generating or scoring candidates. Any restricted product is instantly pruned from the active catalog.
  * **Why Pre-Filtering?** Since new users rely on volatile short-term session signals (`current_cart` additions and live clicks) fused with context, pre-filtering prevents the system from spinning CPU cycles on unsafe inventory, minimizing microservice response latency.

### 3️⃣ Type 3: Loyal / Returning Users (Full Hybrid Engine: Content + Context with Post-Filtering)
* **Scenario:** Engaged for returning customers with a rich profile and extensive historical purchase records (`purchases`).
* **Filtering & Logic (Post-Filtering):**
  * **How it works:** The engine executes heavy matrix dot-products combining the **Content-Based Layer** (Item-to-Item Cosine Similarity matrix `sim_df.pkl` built via text-vectorized TF-IDF product attributes) with the **Contextual Layer** ($H_i, S_i, D_i$) to yield a total hybrid score:
  
    $$score_{final} = w_1 \cdot score_{content} + w_2 \cdot score_{context}$$
    
  * **Why Post-Filtering?** To maintain structural matrix integrity during complex mathematical similarities, health and allergen constraints are applied as a post-processing layer to strip unsafe items right before building the final response payload.

---

## ⚙️ Computational Architecture (Offline Batch vs. Real-Time Pipelines)

To guarantee sub-millisecond API responses and prevent processing bottlenecks, the codebase decouples heavy statistical operations from live API lookups:

### 🧠 1. Offline Pipeline (Heavy Batch Processing)
* **Item Similarity Matrices:** Pairwise text compositions and product descriptions are tokenized and vectorized offline via Scikit-Learn's TF-IDF Vectorizer. Dense similarity calculations are dumped into serialized structures (`sim_df.pkl`).
* **Daily Expiration Indexing ($D_i$):** Product shelf-life data does not fluctuate minute-by-minute. The baseline Anti-Waste emergency matrix ($D_i$) is computed via a **Daily Batch Job (once every 24 hours)**, drastically removing runtime database stress.

### ⚡ 2. Real-Time Pipeline (Low-Latency Online Inference)
* **Live Ingestion:** Upon receiving a REST payload, the online microservice acts as a lightweight lookup layer, ingesting live user attributes (`current_cart`, active hour, active season).
* **On-the-Fly Blending:** It dynamically fetches the pre-computed offline tensors, overlays them with contextual weights, applies the respective pre/post filtering, and sorts the candidate array instantaneously.

---

## ⚖️ Anti-Starvation & Catalog Diversity Logic (The Max-2 Per Category Rule)

Unconstrained hybrid recommenders inherently suffer from **Popularity Bias**. If a returning user has a strong history of purchasing dairy, a naive algorithm will crowd the Top-10 list with 10 different yogurt flavors, leading to the **Famine (Starvation) Phenomenon** for other categories and degrading user experience.

* **Dynamic Categorical Capping:** During the final sorting phase, the inference engine enforces a strict diversification guardrail. It allows a **maximum of 2 items per category** within the final visible layout.
* **Mechanism:** If a category becomes saturated ($\ge 2$), subsequent items from that category are bypassed, forcing the engine to select the next best-performing candidate from an under-represented category.
* **Impact:** This mathematical threshold boosts **Intra-List Diversity** and maximizes **Catalog Coverage**, ensuring alternative supermarket aisle products get optimal visibility.

---

## 📊 Evaluation & System Health Metrics

Model performance, ranking correctness, and list variety are continuously tracked in `evaluation_metrics.py` across major standard RecSys metrics:
* **Precision@K & Recall@K / F1-Score** * **NDCG@K** (Normalized Discounted Cumulative Gain for ranking order quality)
* **Mean Reciprocal Rank (MRR) & Hit Rate**
* **Catalog Coverage & Intra-List Diversity** (Ensuring high asset distribution)

---

## 🛠️ Tech Stack & Dependencies

* **Core Framework:** Flask (REST API)
* **Data Science & ML Engineering:** Python, Pandas, NumPy, Scikit-Learn, Joblib, Pickle
* **Storage Artifacts:** Compressed Python Pickles / serialized joblib models

---
