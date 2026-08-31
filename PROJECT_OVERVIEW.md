# FlyRank Applied Search Intelligence: Complete Project & Status Overview

> **Document Purpose:** Comprehensive summary of the FlyRank ML project — what it is, its core philosophy, the overall roadmap, what has been completed so far (with metrics and data artifacts), and what is currently in progress.  
> **Last Updated:** August 27, 2026  
> **Active Branch:** `my-changes`  
> **Selected Lane:** CTR / Engagement Opportunity Scoring  

---

## 1. What Is This Project?

### Context & Problem Domain
The **FlyRank ML Internship** focuses on **Applied Search Intelligence: Google Search Ranking & Discoverability**. 

In search engine optimization (SEO) and content operations, large publishing networks and businesses produce tens of thousands of pages. Traditional SEO often relies on vague heuristics ("make content longer", "add more keywords") or unverified correlations. 

The goal of this project is to build an **honest, empirical, machine-learning-driven decision support system** that:
1. Ingests large-scale Google Search Console (GSC) performance and Google Analytics 4 (GA4) behavioral data.
2. Identifies specific, high-leverage content underperforming relative to its search visibility (pages with strong impressions and ranking position that fail to capture clicks).
3. Produces a prioritized, actionable queue for editorial teams (e.g., rewriting titles and meta descriptions, improving snippet relevance).

### Core Principles ("Honest ML")
Throughout this project, strict methodological guardrails are enforced:
- **No Circular Measurement / Leakage:** Features must strictly precede labels in time. Label-derived columns (like future clicks or post-event platform flags) are rigorously eliminated.
- **Honest Splits:** Standard random train/test splits leak client signatures and future performance. Evaluation requires **group-aware splits** (holding out entire clients) and **time-aware splits** (training on past months, evaluating on future months).
- **Baseline-First Benchmark:** An ML model must justify its complexity by beating transparent, rule-based heuristics on the exact same split and metrics.
- **Defensive, Scientific Claim Language:** We never claim to "predict Google's ranking algorithm" or make unproven causal assertions. All findings are framed as **observed**, **measured**, **directional**, and **decision-support**.
- **Data Privacy:** Raw client domains, names, URLs, and private search queries are never committed or exposed (CI enforces this).

### The Data Architecture
- **Starter Dataset:** `data/raw/content_refresh_anonymized.csv` — 30,000 anonymized content items across 32 clients with 44 columns covering 90-day search aggregates, engagement metrics, and content metadata.
- **Production Warehouse:** `hf://datasets/FlyRank/internship-warehouse` — full HuggingFace warehouse (~79 million rows from January to June 2026) queried remotely without full local download using **DuckDB**.

---

## 2. What We Are Doing (The Roadmap)

The project follows an 8-week, 10-assignment sequence moving from conceptual framing to a production-grade research paper:

```
[ML-02] Research Question & Lane Selection (Week 1)
   │
[ML-03] Task Formulation & Metric Definition (Week 2)
   │
[ML-04] Warehouse Data Contract & Grain Check (Week 3)
   │
[ML-05] Feature Engineering & Leakage Hunt (Week 3)
   │
[ML-06] Empirical Signal Audit & Hypothesis Testing (Week 4)
   │
[ML-07] Transparent Rule Baseline & Top-20 Review (Week 4)
   │
[ML-08] Honest Model Training (Temporal & Group Split) (Week 5)
   │
[ML-09] Validation Audit & Paper Audit (Week 6)  <─── [CURRENT FOCUS]
   │
[ML-10] Editorial Action Playbook (Week 7)
   │
[ML-11] Capstone Research Paper & Web Deployment (Week 8)
```

### Our Chosen Lane: CTR / Engagement Opportunity Scoring
- **Target Question:** *"Which visible, well-positioned pages are under-capturing clicks relative to what their ranking tier predicts, and should be prioritized for title/meta optimization?"*
- **Primary Decision Maker:** SEO content editors / content leads.
- **Action:** Reviewing snippets, search intent alignment, and rewriting title tags and meta descriptions.
- **Error Costs:**
  - *False Positive:* Editor reviews a page that was already optimal (small operational cost; recoverable).
  - *False Negative:* A genuinely underperforming high-traffic page remains unfixed (silent, ongoing loss of potential clicks and traffic).
- **Core Evaluation Metric:** **Precision@K** (primarily $K=20$ and $K=50$, representing weekly editorial review batch sizes) alongside **Average Precision (AP)** and **ROC-AUC**.

---

## 3. What Has Been Done (Detailed Breakdown)

| Assignment | Card | Notebook | Key Artifacts | Status |
|---|---|---|---|---|
| **ML-02** | Week 1 | `work/notebooks/w01_research_question.ipynb` | Decision framing log | ✅ Completed |
| **ML-03** | Week 2 | `work/notebooks/w02_ml_task_framing.ipynb` | Proxy target & formula | ✅ Completed |
| **ML-04** | Week 3 | `work/notebooks/w03_data_contract.ipynb` | DuckDB HF integration | ✅ Completed |
| **ML-05** | Week 3 | `work/notebooks/w03_feature_leakage_check.ipynb` | 40-feature matrix | ✅ Completed |
| **ML-06** | Week 4 | `work/notebooks/w04_signal_audit.ipynb` | Signal test scorecards | ✅ Completed |
| **ML-07** | Week 4 | `work/notebooks/w04_baseline_score.ipynb` | `baseline_action_score.csv`, `baseline_metrics.json` | ✅ Completed |
| **ML-08** | Week 5 | `work/notebooks/w05_model.ipynb` | `hf_features_dev.parquet`, Model comparison | ✅ Completed |
| **ML-09** | Week 6 | `work/notebooks/w06_validation_audit.ipynb` | Audit notebook | 🟡 **Active Next** |
| **ML-10** | Week 7 | `work/notebooks/w07_action_playbook.ipynb` | Playbook | ⏳ Pending |
| **ML-11** | Week 8 | `work/notebooks/capstone.ipynb` | Capstone paper & report | ⏳ Pending |

---

### Step-by-Step Achievements & Findings

#### ML-02 — Research Question & Decision Framing
- Selected **CTR / Engagement Opportunity Scoring** over content freshness or position-decay lanes.
- Evaluated starter data (30k rows): found **9,759 pages (32.5%)** with high visibility (impressions $\ge 500$, position $\le 20$) but low CTR ($<0.5\%$), and **7,113 pages (23.7%)** with low engagement.
- Defined empirical CTR tier benchmarks: Top 3: 2.76%, Page 1 (pos 4–10): 0.65%, Striking distance (pos 11–20): 0.32%, Page 3–5: 0.22%, Deep ($>50$): 0.15%.

#### ML-03 — ML Task Framing & Proxy Target Engineering
- **Data Quality Catch:** Identified that 1,205 rows with `avg_position == 0` (unranked/no impressions) were mistakenly bucketed into `top_3` by naive $\le 3$ checks. Filtered to 28,795 valid items. Without this fix, Top 3 CTR expectation was distorted to 1.48% (actual: 2.76%).
- Formulated the problem as a **Ranking/Scoring Task** rather than binary classification.
- Engineered 3-step opportunity score:
  $$\text{Opportunity Gap} = \mathbb{E}[\text{CTR} \mid \text{position\_tier}] - \text{CTR}_{\text{page}}$$
  $$\text{Weighted Score} = \text{Opportunity Gap} \times \ln(1 + \text{Impressions})$$
- **Discovered Tier-Domination Bias:** Because Top 3 expected CTR is 2.76% vs 0.65% for Page 1, the top-25 recommendations were completely dominated by Top 3 pages, starving viable striking-distance candidates. This demonstrated why a learned ML model is needed over a rigid rule.

#### ML-04 — Search Intelligence Data Contract (Full Warehouse)
- Connected DuckDB directly to HuggingFace dataset warehouse (`FlyRank/internship-warehouse`, 79M rows across 2026-01 to 2026-06) using secure token access.
- Selected development partition `month = '2026-03'` (9.84M rows, 321,106 distinct visible content items across clients).
- Verified data grain: 0 items crossed clients (`content_hash_id` strictly belongs to 1 `client_hash_id`).
- Proved intentional leakage: adding raw clicks into a score boosted Precision@25 artificially from 0.00 to 0.76 — demonstrating the necessity of strict feature isolation.
- Discovered non-random missingness in `word_count` (missing 37.5% in keyword articles vs 0.09% in comparison articles).

#### ML-05 — Feature Engineering & Leakage Elimination
- Built a 40-feature representation (14 numerical, 26 one-hot encoded).
- Ran three-tier leakage audits:
  1. Correlation check with target ($r = -0.74$ with raw CTR confirmed, isolating target).
  2. Window correlation: checked 30-day sub-windows.
  3. Product/post-event flags: discovered and pruned `provider_used` (71.8% missing) and `model_used` (19.9% missing).

#### ML-06 — Empirical Signal Audit
- **Mini-Test 1 (Impression Volume vs Noise):** Proved that pages with $<50$ impressions have high zero-CTR variance; verified volume thresholds are non-negotiable.
- **Mini-Test 2 (Search Intent vs CTR):** Evaluated navigational vs informational vs transactional CTR distributions.
- **Mini-Test 3 (Word Count vs CTR):** Found word count alone has a weak, non-linear relationship with CTR, invalidating the popular SEO myth that "longer always equals higher CTR".
- **Flag-Linked Test:** Audited the `needs_ctr_fix` indicator against empirical opportunity gaps.

#### ML-07 — Baseline Action Score & Ranked Queue
- Implemented a transparent, rule-based heuristic with 8 structured reason codes (e.g., `top3_ctr_underperformer`, `page1_ctr_underperformer`, `low_impression_volume`).
- Exported production artifacts:
  - `work/outputs/baseline_action_score.csv` (5.3 MB): Actionable ranked queue assigning items to `rewrite_title_and_meta`, `review_snippet_and_intent`, `monitor_ctr`, or `no_action`.
  - `work/outputs/baseline_metrics.json`: Detailed distribution metrics.
- Conducted manual qualitative review of the Top-20 items to verify editorial credibility.

#### ML-08 — Honest Model Training & Capstone Modeling Lane
- Designed an uncompromised **Time-Aware + Client-Grouped Split**:
  - **Feature History Window:** Q1 (January–March 2026) — 60,088 aggregated items cached in `work/outputs/hf_features_dev.parquet`.
  - **Future Outcome Window:** Q2 (April–June 2026) — target label `ctr_improved = 1` if future CTR achieved $\ge 10\%$ relative lift over baseline.
  - **Split Strategy:** `GroupShuffleSplit` on `client_hash_id` (20% held-out test clients, zero overlap).
- Benchmark Results on Held-Out Test Clients:
  | Method | Base Rate | Precision@20 | Precision@50 | Avg Precision | ROC-AUC |
  |---|---|---|---|---|---|
  | **Base Rate (Random)** | 62.0% | 62.0% | 62.0% | 62.0% | 0.500 |
  | **Rule Baseline (Week 4)** | 62.0% | 45.0% | 42.0% | 66.8% | 0.613 |
  | **Logistic Regression** | 62.0% | **100.0%** | **100.0%** | **93.8%** | **0.880** |
  | **Random Forest** | 62.0% | **100.0%** | **100.0%** | **94.8%** | **0.902** |
- **Error Analysis:** Top 3 and Page 1 pages had lower prediction accuracy (~72%–77%) than deep pages (>93%), diagnosing that top SERP positions suffer from external SERP feature volatility (AI Overviews, featured snippets, PAA) not captured in historical click aggregates.

---

## 4. What Is Happening Right Now?

### The Current Focus: ML-09 (`w06_validation_audit.ipynb`)
We have completed Week 5 (ML-08) and are stepping into **Week 6 (ML-09): Validation and Research Claim Audit**.

The objective of this assignment is to subject both external research and our own model to rigorous audit:
1. **Audit Two Findings from FlyRank's Published Research Paper** (`flyrank-seo-research-march-2026.pdf`):
   - *Finding #1 ("Anatomy of Growing Content"):* Scrutinize whether longer content causes growth or if this is observational correlation with temporal overlap between `trend_pct` sub-windows.
   - *Finding #4 ("The Freshness Multiplier"):* Investigate survivorship bias and unobserved editorial confounding in the reported 283:1 growth ratio for refreshed content.
2. **Audit Our Own Model Under Honest Splits:**
   - Document the contrast between naive random splits and our client-holdout time-split.
3. **Leakage & Error Audit:**
   - Verify feature timing and explain why top-tier pages produce false negatives.
4. **Rewrite Claims with Defensive Language:**
   - Enforce rigorous, cautious terminology (*observed*, *measured*, *directional*, *decision-support*).

### What Comes Next:
- **ML-10 (`w07_action_playbook.ipynb`):** Translating model predictions and priority scores into an operational playbook and editorial SLA guidelines.
- **ML-11 (`capstone.ipynb` & Research Paper):** Finalizing the capstone report (`work/capstone_report.md`), producing publication-ready visual assets, and deploying the static paper to GitHub Pages.
