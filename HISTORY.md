# HISTORY.md — FlyRank Internship Decision & Action Log

> **Purpose:** every key decision, finding, and action is recorded here with its exact reason.
> Update this file at the end of every notebook or task so context is always available.
>
> **Rule:** add a new `---` section per notebook/task. Never delete old entries.

---

## ML-02 — Research Question and Provisional Lane
**Notebook:** [`work/notebooks/w01_research_question.ipynb`](work/notebooks/w01_research_question.ipynb)
**Completed:** ~Jul 2026 (uploaded via GitHub)
**Status:** ✅ Submitted

### Lane chosen
**CTR / Engagement Opportunity Scoring**

> *Why this lane?* The dataset contains direct CTR and engagement signals alongside position
> data, making it possible to compute a position-adjusted gap without external labels.

### Decision: Decision framing
| Dimension | Decision | Reason |
|---|---|---|
| **Decision** | Which visible, well-positioned pages are under-capturing clicks or engagement relative to what their position tier predicts? | Focuses on actionable, measurable gaps rather than vague "content quality" |
| **Actor** | SEO editor / content owner | They receive the ranked list and review title, meta, snippet for highest-priority pages |
| **False positive cost** | Wasted review time on a fine page | Low severity, recoverable |
| **False negative cost** | Real under-performing page goes unnoticed → ongoing lost clicks/engagement | Higher severity, silent |

### Key numbers found (from starter dataset, 30k rows)
| Finding | Value |
|---|---|
| Low-CTR visible pages (impressions ≥ 500, avg_position ≤ 20, ctr < 0.5%) | **9,759 / 30,000 (32.5%)** |
| Low-engagement visible pages (sessions ≥ 30, engagement_rate or scroll_rate < 30) | **7,113 / 30,000 (23.7%)** |
| Mean CTR by position tier | top_3=2.76, page_1=0.65, striking=0.32, page_3_5=0.22, deep=0.15 |

### Claim boundaries set
- **Can claim:** page-level CTR gap vs. tier average is an observed, position-adjusted, directional signal.
- **Cannot claim:** gap is caused by a weak title/meta (could be SERP features, intent mismatch, seasonality). Cannot claim editing will cause CTR to improve without an experiment.

---

## ML-03 — Frame the Lane as an ML Task
**Notebook:** [`work/notebooks/w02_ml_task_framing.ipynb`](work/notebooks/w02_ml_task_framing.ipynb)
**Completed:** Jul 27 2026 (commits: 3249b1e → fd4deec → f6b6606)
**Status:** ✅ Submitted

### Decision: Task type → Ranking/Scoring (not classification)
| Option | Decision | Reason |
|---|---|---|
| Binary classification | ❌ Rejected | "Opportunity" is not yes/no; cutoffs collapse the continuous gap |
| Ranking / Scoring | ✅ Chosen | Output is an ordered list — reviewers need priority rank, not a flag |
| Clustering | ❌ Rejected | No requirement to group; need a per-item score |

### Decision: Target/Proxy construction (3-step engineered score)

**Step 0 — Critical data quality fix:**
```
PROBLEM: position_tier == "no_data" never appears in raw CSV.
FOUND:   1,205 rows have avg_position == 0 but bucketed into "top_3"
         (tier rule avg_position <= 3 unintentionally captures 0).
FIX:     Filter on avg_position > 0 directly (not on the string label).
EFFECT:  30,000 → 28,795 valid rows.
         Without fix: expected_ctr(top_3) = 1.48. With fix: 2.76 (almost double).
         Every downstream gap for top_3 pages depends on this.
```

**Step 1 — Raw opportunity gap:**
```python
expected_ctr = valid.groupby("position_tier")["ctr"].transform("mean")
opportunity_gap = expected_ctr - ctr   # positive = underperforming vs. tier peers
```

**Step 2 — Weight by traffic volume (log-scale):**
```python
weighted_score = opportunity_gap * np.log1p(impressions_90d)
```
> Why log1p: raw impressions span tens→hundreds of thousands; gap only ranges ~0–3.
> Direct multiplication lets volume dominate. log1p compresses range while preserving order.
> log1p also handles impressions_90d == 0 safely.

### Key finding: Tier-domination bias discovered
```
Top-25 by weighted_score are ALL top_3 pages.
Reason: max possible gap for any tier = expected_ctr(tier) at ctr=0.
        top_3 ceiling (2.76) >> all other tiers combined.
        deep-tier pages cannot surface no matter their traffic volume.
Status: OPEN — not resolved in this notebook.
Planned fix: tier-stratified ranking / per-tier quotas / learned model.
```

### Decision: Success metric → Precision@K (K ≈ 25)
| Choice | Reason |
|---|---|
| K = 25 | Assumption: ~5 pages/day × 5-day week. **Stated as assumption, not measured fact.** |
| Precision@K (not ROC AUC) | "Did top K include real opportunities?" — a ranking precision question |
| Not computable from static dataset | Requires editor review labels or a forward time window. precision_at_k() function written and ready; will run once labels exist. |

### Unit of analysis
One row = one content page, observed over 90 days, with valid ranking signal (avg_position > 0).
28,795 valid units. Core columns: `content_id`, `client_id`, `position_tier`, `ctr`,
`impressions_90d`, `engagement_rate`, `scroll_rate`, `content_type`, `main_intent`.

### Decision: Why ML beats a fixed rule
Fixed rule (`weighted_score`) exposed its own tier-domination bias. Adding more factors
(content_type, main_intent, freshness) requires hand-coding every interaction — combinatorial
explosion. An ML model learns those interactions from data directly.

---

## ML-04 — Search Intelligence Data Contract
**Notebook:** [`work/notebooks/w03_data_contract.ipynb`](work/notebooks/w03_data_contract.ipynb)
**Completed:** Aug 7–8 2026 (commits: daa24b7 → 06889f9 → 8205667)
**Status:** ✅ Submitted

### Data source: Full warehouse (Hugging Face, ~79M rows)
Switched from starter CSV (30k rows) to live warehouse via DuckDB + HuggingFace.
```python
REL = 'hf://datasets/FlyRank/internship-warehouse'
con = duckdb.connect()
con.execute(f"CREATE OR REPLACE SECRET hf (TYPE huggingface, TOKEN '{HF_TOKEN}')")
```
HF token via `getpass`, never hardcoded in notebook.

### Decision: Development month = 2026-03 (not the sample/June)
| Month | Reason to avoid |
|---|---|
| _sample (June 2026) | Natural outcome window → leaks into any future past→future label |
| **2026-03** | ✅ Mid-panel month, no leakage risk for current pass |

### Grain verification: ✅ PASSED
```
Query: content items spanning >1 client_hash_id in 2026-03
Result: 0
Conclusion: one content item = one client. Grain holds without ambiguity.
```

### Window / scale verification
```
Total rows in 2026-03:            9,841,378
Visible rows:                     9,532,718  (≈96.9%)
Distinct visible content items:     321,106
Date span: 2026-03-01 → 2026-03-31 ✅ (full month, no gaps)
```

### Field classification
| Category | Fields |
|---|---|
| **Context** (never fed to model) | `content_hash_id`, `client_hash_id`, `report_date`, `month`, `is_published`, `is_deleted` |
| **Label / proxy** | `ctr = gsc_clicks / gsc_impressions` (raw; not position-adjusted in this pass) |
| **Features** | `gsc_avg_position`, `gsc_impressions`, `content_type`, `word_count`, `content_age_days` (derived) |
| **Excluded** | `gsc_clicks` (used to build label — direct leakage if used as feature); GA4 engagement columns (deferred: need `ga4_data_available IS TRUE` filter first) |

### Feature frame: 5 features, 321,106 rows
```sql
avg_position     = AVG(gsc_avg_position) over 2026-03
impressions      = SUM(gsc_impressions) over 2026-03
clicks           = SUM(gsc_clicks) over 2026-03
ctr              = CASE WHEN impressions > 0 THEN clicks / impressions ELSE NULL END
content_age_days = DATE_DIFF('day', content_created_date, last_report_date)
```
After minimum-volume filter (impressions ≥ 50): **116,063 rows**.

### Leakage demonstration (deliberate, then removed)
```
Honest score (impressions / avg_position only):        Precision@25 = 0.000
Leaky score  (honest + ctr × 1,000,000):              Precision@25 = 0.760
```
> ⚠️ The 0.76 jump is entirely artificial — circular measurement, not signal.
> Leaky column dropped immediately. Honest 0.000 is the real baseline for this pass.

### Data limit found: word_count missingness is NOT random
```
Overall missing: 32.9% of visible content items

By content_type:
  comparison article  →  0.09% missing
  feedly article      →  5.7%  missing
  keyword article     → 37.5%  missing  ← most common type
```
> **ACTION (future):** Never fillna(0) for word_count — that silently encodes a content_type
> signal. Must add a `has_word_count` boolean flag alongside any imputed value.

---

## Open Issues Carried Forward

| Issue | Opened in | Status | Notes |
|---|---|---|---|
| Tier-domination bias in weighted_score | ML-03 | 🔴 Open | top_3 always dominates top-K. Fix: tier-stratified ranking or learned model |
| Precision@K = 0.000 (honest baseline) | ML-04 | 🔴 Open | Raw impressions/position doesn't predict top-CTR pages. Needs better features |
| GA4 engagement columns deferred | ML-04 | 🟡 Deferred | Need `ga4_data_available IS TRUE` filter first |
| word_count requires `has_word_count` flag | ML-04 | 🟡 Deferred | Before using word_count as model feature |
| Precision@K not measurable from static data | ML-03 | 🟡 Blocked | Needs editor review labels or forward time window |
| opportunity_gap not yet position-adjusted in warehouse pass | ML-04 | 🟡 Deferred | Contract uses raw CTR as proxy; tier-adjusted gap is future work |

---

## ML-05 — Feature Vector and Leakage/Privacy Check
**Notebook:** [`work/notebooks/w03_feature_leakage_check.ipynb`](work/notebooks/w03_feature_leakage_check.ipynb)
**Completed:** Aug 10 2026 (commit: a2b883a → f693b90)
**Status:** ✅ Completed

### Key findings & actions
- Built 40-feature representation (14 numeric, 26 one-hot encoded) from 28,795 valid items.
- Tested correlation between features and `weighted_score`: raw CTR showed $r = -0.74$, isolating target.
- Sub-window analysis showed weak correlation with 30-day metrics.
- Excluded post-creation metadata flags `provider_used` (71.8% missing) and `model_used` (19.9% missing) to prevent leakage.

---

## ML-06 — Signal Audit: Do the Flags Hold?
**Notebook:** [`work/notebooks/w04_signal_audit.ipynb`](work/notebooks/w04_signal_audit.ipynb)
**Completed:** Aug 2026 (commit: a72e2ac)
**Status:** ✅ Completed

### Key findings & actions
- **Mini-Test 1 (Impression Volume vs Zero-CTR Noise):** Confirmed pages with impressions $<50$ are dominated by zero-CTR noise. Minimum volume cutoff is essential.
- **Mini-Test 2 (Search Intent vs CTR):** Confirmed commercial/transactional queries exhibit higher variance and different baseline CTR than informational queries.
- **Mini-Test 3 (Word Count vs CTR):** Weak non-linear relationship observed; refutes the heuristic that longer content inherently yields higher CTR.
- **Flag-Linked Test:** Confirmed `needs_ctr_fix` flag aligns with high-opportunity gaps in positions 1–20.

---

## ML-07 — Baseline Action Score and Top-20 Review
**Notebook:** [`work/notebooks/w04_baseline_score.ipynb`](work/notebooks/w04_baseline_score.ipynb)
**Completed:** Aug 2026 (commit: 477b807)
**Status:** ✅ Completed

### Key findings & actions
- Built a transparent rule-based baseline assigning pages to 4 action categories (`rewrite_title_and_meta`, `review_snippet_and_intent`, `monitor_ctr`, `no_action`) across 8 explicit reason codes.
- Exported artifacts: `work/outputs/baseline_action_score.csv` (5.3 MB, 30,000 items) and `work/outputs/baseline_metrics.json`.
- Performed manual top-20 review: confirmed top picks are actionable top-3/page-1 underperformers with high impression volume.

---

## ML-08 — Capstone Modeling Lane
**Notebook:** [`work/notebooks/w05_model.ipynb`](work/notebooks/w05_model.ipynb)
**Completed:** Aug 2026
**Status:** ✅ Completed

### Key findings & actions
- **Time-split:** Feature window Q1 (Jan–Mar 2026) vs Label window Q2 (Apr–Jun 2026) with volume floor $\ge 100$ impressions. Cached 60,088 items to `work/outputs/hf_features_dev.parquet`.
- **Target:** `ctr_improved` (binary outcome indicating $\ge 10\%$ relative CTR lift in Q2 over Q1). Base rate = 62.0%.
- **Validation Split:** Client-holdout (`GroupShuffleSplit` on `client_hash_id`, 20% test clients).
- **Results on Test Clients:**
  - Base Rate: 62.0%
  - Week 4 Rule Baseline: Precision@20 = 45.0%, Precision@50 = 42.0%, ROC-AUC = 0.613
  - Logistic Regression: Precision@20 = 100.0%, Precision@50 = 100.0%, ROC-AUC = 0.880
  - Random Forest: Precision@20 = 100.0%, Precision@50 = 100.0%, ROC-AUC = 0.902
- **Error Analysis:** Top 3 and Page 1 show highest error rates (accuracy 72%–77%, FN rate 18%–22%) due to external SERP features (AI Overviews, snippets) not captured in aggregates. Deep tiers show >93% accuracy.

---

## ML-09 — Validation & Research Claim Audit
**Notebook:** [`work/notebooks/w06_validation_audit.ipynb`](work/notebooks/w06_validation_audit.ipynb)
**Completed:** Aug 2026
**Status:** ✅ Completed

### Key findings & actions
- Audited two empirical findings from the FlyRank research paper (`flyrank-seo-research-march-2026.pdf`): word count vs. impression growth and AI session sparsity.
- Verified client-grouped time-split properties on Week-5 model: 0% overlap between train and test client domains.
- Documented hard failure modes: False Positives on high-headroom zero-click SERPs (AI Overviews) and False Negatives on low-volume striking distance pages.
- Rewrote research claims to adhere strictly to honest, non-causal, decision-support terminology.

---

## ML-10 — Content Action Playbook
**Notebook:** [`work/notebooks/w07_action_playbook.ipynb`](work/notebooks/w07_action_playbook.ipynb)
**Completed:** Sep 2026
**Status:** ✅ Completed

### Key findings & actions
- Scored all 60,088 pages using composite priority scoring formula: $\text{Priority Score} = \text{rf\_prob} \times \text{opportunity\_gap} \times \ln(1 + \text{total\_impressions})$.
- Assigned 4 action tiers across all content:
  - `rewrite_title_and_meta`: 2,883 pages (4.8%) — immediate sprint triage.
  - `review_snippet_and_intent`: 10,038 pages (16.7%) — Page 1 review.
  - `monitor_ctr`: 16,370 pages (27.2%) — observation watchlist.
  - `no_action`: 30,797 pages (51.3%) — benchmark performers left untouched.
- Defined monitoring and retrain triggers: alert if production CTR improvement rate drops below 45% or Top 3 benchmark drifts outside $[2.35\%, 3.18\%]$.
- Exported production artifacts to `work/outputs/`: `action_queue_top100.csv`, `action_distribution.png`, `priority_score_by_tier.png`, `rf_prob_distribution.png`, `playbook_summary.json`.

---

## ML-11 — Capstone Report & Research Paper
**Notebook:** [`work/notebooks/capstone.ipynb`](work/notebooks/capstone.ipynb)
**Report:** [`work/capstone_report.md`](work/capstone_report.md)
**Deployed Page:** [`docs/index.html`](docs/index.html) (`https://moath177.github.io/flyrank/`)
**Completed:** Sep 2026
**Status:** ✅ Completed

### Key findings & actions
- Filled and executed `work/notebooks/capstone.ipynb` top-to-bottom across all 7 canonical research paper sections.
- Authored the comprehensive Capstone Report `work/capstone_report.md` fulfilling all 8 axes of the evaluation rubric and claims checklist.
- Built and published the standalone public Research Paper web page (`docs/index.html`) deployed to GitHub Pages.
- Registered the deployed research paper URL in `submission/paper_url.txt`.

