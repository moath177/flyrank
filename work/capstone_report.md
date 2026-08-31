# Capstone Report — CTR / Engagement Opportunity Scoring

- **Author:** FlyRank ML Intern (moath177)
- **Lane:** CTR / Engagement Opportunity Scoring
- **Repo:** https://github.com/moath177/flyrank
- **Date:** September 2026

---

## 1. Problem framing

### Decision Supported
Enterprise search and content teams frequently manage tens of thousands of published articles, guides, and commercial landing pages. A critical operational challenge is identifying which high-visibility pages are significantly underperforming in organic Click-Through Rate (CTR) and would benefit most from snippet optimization (title tag, meta description, and schema markup adjustments). 

Manual editorial auditing across large enterprise catalogs is cost-prohibitive. This system provides an automated, machine-learned decision-support engine that scores and ranks underperforming content, allowing editorial teams to allocate finite writing bandwidth to pages with the highest expected organic click recovery.

### Unit of Analysis & System Output
* **Unit of Analysis:** Unique page item aggregated across a discrete monthly observation window (`content_hash_id` × 30-day window).
* **System Output:** A continuous composite Priority Score ($	ext{rf\_prob} 	imes 	ext{opportunity\_gap} 	imes \ln(1 + 	ext{impressions})$) mapping directly to a ranked action queue with discrete editorial triage tags (`rewrite_title_and_meta`, `review_snippet_and_intent`, `monitor_ctr`, `no_action`).
* **Human Action:** Writers and SEO editors review the top of the priority queue, inspect search intent and current SERP layout, and deploy revised title tags and meta descriptions.

### Cost of a Wrong Call
* **False Positive (Flagging a page that cannot improve):** Editorial hours are wasted rewriting snippet metadata on pages where low CTR is structural (e.g. suppressed by dominant AI Overviews or navigational intent).
* **False Negative (Missing a high-opportunity page):** Potential organic click recovery is foregone, leaving high-ranking content chronically under-harvested.
* **Over-Editing Stable Content:** Unnecessary revisions to content already performing at or above position benchmark risk disrupting stable snippet CTR.

### Why Machine Learning Helps
Heuristic rules (such as ranking purely by impression volume or raw CTR distance) fail because CTR potential is highly non-linear and conditional on ranking tier, baseline traffic volatility, and starting CTR scale. Supervised classification with non-linear tree ensembles captures subtle multi-feature interactions, achieving near-perfect precision at the top of the queue while filtering out false opportunities.

---

## 2. Data safety

### Data Release & Provenance
All analyses utilize the **FlyRank ML Internship Dataset** hosted in the Hugging Face gated warehouse (`hf://datasets/FlyRank/internship-warehouse`). We utilize the partitioned `fact_content_daily_performance` table covering daily search telemetry.

### Cohort Definition & Exclusions
* **Observation Windows:**
  * **Feature Window (History):** January 1 – January 31, 2026 (31 days).
  * **Label Window (Future Outcome):** February 1 – February 28, 2026 (28 days).
* **Volume Floor:** Content items with fewer than 100 cumulative impressions in either window are strictly excluded. Low-volume pages introduce extreme binomial variance in CTR calculations.
* **Activity Filter:** Pages with zero active days or zero recorded impressions are pruned.
* **Final Cohort:** 60,088 verified page observations across 28 distinct enterprise clients.

### Deliberate Exclusions & Leakage Prevention
* **Target-Derived Columns Excluded:** Target metrics (`label_ctr`, `label_impressions`, `ctr_improved`, `trend_direction`, `trend_pct`) are derived exclusively from the February label window and are strictly isolated from the feature matrix.
* **Pseudonymous Identifiers:** Client IDs (`client_hash_id`) and content identifiers (`content_hash_id`) are cryptographic hashes. They are used solely for grouping and validation partitions, never as predictive features.
* **Zero Public Exposure:** No client domain names, proprietary query terms, unhashed URLs, or internal system credentials appear anywhere in the repository, notebooks, or exported artifacts.

---

## 3. Baseline

### Baseline Formulation (Week 4 Rule Heuristic)
We evaluate our machine learning models against the standard heuristic baseline established during Week 4:
$$	ext{baseline\_score} = 	ext{opportunity\_gap} 	imes \ln(1 + 	ext{total\_impressions})$$
where $	ext{opportunity\_gap} = \max(0, 	ext{tier\_expected\_ctr} - 	ext{feat\_ctr})$.

### Baseline Performance on Held-Out Test Set (6 Clients, 15,970 Pages)
* **Base Rate (Random Selection):** 62.0%
* **Baseline Precision@20:** 45.0%
* **Baseline Precision@50:** 42.0%
* **Baseline Average Precision (AP):** 66.8%
* **Baseline ROC-AUC:** 0.613

### Why the Heuristic Fails
The heuristic baseline performs *worse than random guessing* at Precision@20 (45.0% vs. 62.0% base rate). Because it multiplies raw gap by log-impressions, it aggressively elevates massive-volume Page 1 queries where click depression is caused by external SERP layout features (e.g. AI Overviews, Featured Snippets, Knowledge Panels) rather than improvable snippet copy.

---

## 4. Model / analysis

### Method Selection
We evaluated two model architectures:
1. **Logistic Regression (L2-Regularized):** Linear benchmark with `StandardScaler` pipeline for feature interpretability.
2. **Random Forest Classifier (Primary Model):** Non-linear ensemble (`n_estimators=100`, `max_depth=6`, `min_samples_leaf=20`, `random_state=42`) designed to capture non-linear feature interactions without overfitting to client-specific noise.

### Target Definition
A page is labeled positive (`ctr_improved = 1`) if its observed February click-through rate improves by at least 10% relative to its January baseline:
$$	ext{ctr\_improved} = \mathbb{I}\left(rac{	ext{clicks}_{	ext{Feb}}}{	ext{impressions}_{	ext{Feb}}} \ge 1.10 	imes rac{	ext{clicks}_{	ext{Jan}}}{	ext{impressions}_{	ext{Jan}}}ight)$$

### Exact Feature Set (Strictly Jan 2026 Historical Signals)
1. `log_impressions`: Log-transformed impression volume (`log1p(total_impressions)`).
2. `feat_ctr`: Baseline January CTR (`total_clicks / total_impressions`).
3. `avg_position`: Mean organic SERP ranking position across all query impressions.
4. `days_active`: Number of days with active impressions during January.
5. `opportunity_gap`: Headroom beneath the position tier expected CTR (`tier_expected_ctr - feat_ctr`).
6. `ctr_trend`: Linear trajectory of daily CTR within January.

---

## 5. Evaluation

### Validation Design (Client-Grouped Split)
To evaluate generalization to unseen client domains and prevent domain-level data leakage, we partitioned the 60,088 pages using `GroupShuffleSplit(test_size=0.2, random_state=42, groups=client_hash_id)`:
* **Train Partition:** 22 clients (44,118 pages, positive base rate: 57.9%).
* **Test Partition:** 6 held-out clients (15,970 pages, positive base rate: 62.0%).
* **Client Overlap:** 0 clients (0.0% leakage).

### Empirical Model vs. Baseline Comparison Table

| Method | Base Rate | Precision@20 | Precision@50 | Avg Precision | ROC-AUC |
|---|---|---|---|---|---|
| **Base Rate (Random)** | 62.0% | 62.0% | 62.0% | 62.0% | 0.500 |
| **Rule Baseline (Week 4)** | 62.0% | 45.0% | 42.0% | 66.8% | 0.613 |
| **Logistic Regression** | 62.0% | 100.0% | 100.0% | 93.8% | 0.880 |
| **Random Forest (Final)** | **62.0%** | **100.0%** | **100.0%** | **94.8%** | **0.902** |

### Error Analysis & Hard Cases
1. **False Positives (Predicted Lift, CTR Stagnated):** Occurs on high-volume Page 1 queries with low initial CTR. The model occasionally anticipates lift due to massive gap, but clicks remain flat due to unobserved zero-click SERP features (AI Overviews).
2. **False Negatives (Predicted Stagnation, CTR Lifted):** Occurs on low-volume striking distance pages with high starting CTR where a modest raw click increase produces a disproportionately large percentage gain.
3. **Position Tier Error Gradient:** Accuracy is highest on deep positions (>93% accuracy) and lowest on Page 1 / Top 3 (72–77% accuracy) due to intense SERP feature volatility at the head of the search results.

---

## 6. Interpretation

### Feature Importance Hierarchy (Random Forest)
1. **`feat_ctr` (57.1%):** The primary driver. Starting CTR determines mathematical headroom for a +10% relative gain.
2. **`opportunity_gap` (22.5%):** Distance beneath tier benchmark; confirms whether low CTR represents an anomaly or expected tier behavior.
3. **`ctr_trend` (14.6%):** Existing month-internal momentum provides positive confirmation of organic trajectory.
4. **`log_impressions` (3.9%):** Volume weight ensuring stability.
5. **`avg_position` (1.6%):** Ranking tier context.
6. **`days_active` (0.3%):** Basic activity confirmation.

### Permutation Importance Verification
Permutation feature importance confirms that `feat_ctr` causes the steepest drop in test accuracy (mean drop: 0.264), followed by `opportunity_gap` (mean drop: 0.011).

---

## 7. Recommendation

### Composite Priority Scoring
To translate model inference into an actionable queue, all 60,088 pages are scored using:
$$	ext{Priority Score} = 	ext{rf\_prob} 	imes 	ext{opportunity\_gap} 	imes \ln(1 + 	ext{total\_impressions})$$

### Action Allocation Breakdown
* **`rewrite_title_and_meta` (2,883 pages / 4.8%):** Primary sprint action. High-confidence ($	ext{rf\_prob} \ge 0.70$) in Top 3 / Striking distance with $	ext{gap} > 0.50	ext{ pp}$.
* **`review_snippet_and_intent` (10,038 pages / 16.7%):** Page 1 content with $	ext{rf\_prob} \ge 0.70$ and $	ext{gap} > 0.30	ext{ pp}$; requires pre-rewrite audit of SERP features.
* **`monitor_ctr` (16,370 pages / 27.2%):** Moderate confidence candidates placed on automated watchlists.
* **`no_action` (30,797 pages / 51.3%):** Performing at/above benchmark; left untouched to avoid disruption.

### Editorial Operating Protocol
1. **Pre-Action Checklist:** Verify page is live, confirm no rewrites occurred during the observation window, inspect current SERP for AI Overviews, and verify search intent alignment.
2. **No-Go List:** Never perform unreviewed batch rewrites, never score pages with <100 impressions, never apply this model to branded/navigational queries, and never edit `no_action` pages to chase higher vanity scores.

---

## 8. Reproducibility

### Environment & Dependencies
* Python 3.10+ (tested on Python 3.14 / Linux x86_64)
* Core libraries: `duckdb==1.5.5`, `pandas==3.0.5`, `scikit-learn==1.9.0`, `pyarrow==25.0.1`, `matplotlib==3.11.1`, `scipy==1.18.0`

### Execution Instructions
```bash
# 1. Clone repository and activate environment
git clone https://github.com/moath177/flyrank.git
cd flyrank
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Run the full Capstone pipeline top-to-bottom
python3 -m nbconvert --to notebook --execute work/notebooks/capstone.ipynb --inplace
```

### Artifact Manifest
* `work/outputs/action_queue_top100.csv` (28 KB): Top 100 prioritized content recommendations.
* `work/outputs/action_distribution.png` (39 KB): Publication Figure 1.
* `work/outputs/priority_score_by_tier.png` (38 KB): Publication Figure 2.
* `work/outputs/rf_prob_distribution.png` (42 KB): Publication Figure 3.
* `work/outputs/playbook_summary.json` (1.2 KB): Summary metrics for web publication.

---

### Methodological Claims Checklist
- [x] Language uses careful epistemic words: *observed, measured, directional, decision-support*.
- [x] Base rates reported alongside Precision@20/50 and ROC-AUC.
- [x] Zero causal claims asserted without experimental randomized intervention.
- [x] Zero claims of reverse-engineering Google's search algorithms.
- [x] Zero client-identifying details or unhashed URLs.
- [x] Numbers verified against clean top-to-bottom execution.

---

## 9. Appendix: 5-Minute Showcase Demo Outline & Shareable Cuts

### 5-Minute Demo Presentation Structure
* **0:00 – 1:00 (Problem Framing):** Enterprise websites have tens of thousands of URLs with constrained writer bandwidth. How do editorial teams decide which 25–50 pages to rewrite this sprint?
* **1:00 – 2:00 (Data & Clean Split):** Evaluated on 60,088 page-month records across 28 enterprise clients from the FlyRank warehouse with a strictly client-grouped holdout split (0% leakage).
* **2:00 – 3:00 (The Core Finding):** Traditional heuristic rules collapse to 45% Precision@20 by promoting zero-click head queries (AI Overviews). The Random Forest model achieves 100% Precision@20 and 0.902 ROC-AUC by capturing non-linear starting CTR headroom.
* **3:00 – 4:00 (Action Playbook):** Composite priority score isolates 2,883 high-confidence pages (4.8%) with plain-English reason codes and pre-action checklists.
* **4:00 – 5:00 (Epistemic Limits & Wrap-Up):** Observational decision-support tool with transparent boundaries. Live deployed paper at `https://moath177.github.io/flyrank/`.

### Shareable Cuts
1. **Technical Social Post:**
   > Why most SEO prioritization rules fail: sorting by `(Expected CTR - Actual CTR) * Impressions` achieved only 45% Precision@20 across 60k enterprise search records because it over-indexes on zero-click SERPs. A leakage-free Random Forest classifier trained on FlyRank telemetry achieved 100% Precision@20 and 0.902 ROC-AUC by learning starting CTR headroom. Full paper: https://moath177.github.io/flyrank/
2. **3-Sentence Employer-Facing Summary:**
   * **What I built:** A machine-learned search intelligence decision-support engine that scores and prioritizes enterprise content URLs for organic snippet metadata optimization.
   * **On what data:** 60,088 page-month observations across 28 anonymized enterprise client domains from the FlyRank search warehouse, evaluated under a strict client-grouped holdout partition.
   * **What it showed:** The Random Forest classifier achieved 100.0% Precision@20 and an ROC-AUC of 0.902 (vs 45.0% and 0.613 for heuristic baselines), isolating an actionable queue of 2,883 high-leverage pages flagged for immediate editorial triage.
