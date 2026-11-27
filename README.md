# Tata Play Subscriber Retention & Support Copilot

## Overview

This repository implements the four-part capstone as one connected Python project: a trusted data foundation, leakage-aware churn prediction, subscriber segmentation and spend modelling, and a support/retention copilot. All customer, recharge, ticket and policy data are synthetic and fictional. The Tata Play name frames the scenario only; no real Tata Play customer data, policy or pricing is represented. All money values are INR.

The supplied `generate_data.py` and `mock_llm.py` are included unmodified. All model features are available by the 2026-03-31 prediction cutoff; `closure_request_logged` and `next_cycle_recharge_inr` are excluded from churn features because they are not known at prediction time.

## Reproduce

Use Python 3.11 or 3.12 and SQLite 3.25 or newer. On Windows PowerShell, run these commands from the repository root (on macOS/Linux, activate with `source .venv/bin/activate`).

1. `python -m venv .venv`
2. `.\.venv\Scripts\Activate.ps1` (Windows PowerShell)
3. `python -m pip install --upgrade pip`
4. `pip install -r requirements.txt`
5. `pip freeze > environment-freeze.txt `
6. `python generate_data.py`
7. `python part_a.py`
8. `python part_b.py`
9. `python part_c.py`
10. `python part_d.py`
11. `python copilot.py`
12. `python update_readme.py` (after Parts D and the 15 briefs finish successfully; inserts the measured Part D tables below)

The data generator's final line must be `generated: subscribers=2415 recharges_raw=13055 tickets=150 policy_files=5`. The first run of Part D downloads the free `all-MiniLM-L6-v2` sentence-transformer model and caches it. No API key or hosted LLM is required. All language-model calls use `mock_llm.py` by default. For optional live mode only, set `LLM_MODE=live` and `OPENROUTER_API_KEY` in the environment; never put a key in source control. The scripts write the required clean data, SQLite database and model/output files under `data/` and `outputs/`.

## Part A

### Raw data audit

Raw subscribers shape: **2,415 × 17**. Raw recharges shape: **13,055 × 6**. The following tables report each raw column's pandas dtype and missing-value count and percentage.

**Subscribers**

| **Column**        | **dtype** | **Missing** | **Missing %** |
| ----------------------- | --------------- | ----------------- | ------------------- |
| subscriber_id           | object          | 0                 | 0.00%               |
| city                    | object          | 0                 | 0.00%               |
| plan_id                 | object          | 0                 | 0.00%               |
| tenure_months           | int64           | 0                 | 0.00%               |
| age                     | float64         | 99                | 4.10%               |
| payment_mode            | object          | 116               | 4.80%               |
| auto_renew              | object          | 0                 | 0.00%               |
| add_on_count            | int64           | 0                 | 0.00%               |
| monthly_recharge_inr    | int64           | 0                 | 0.00%               |
| hours_watched_per_week  | float64         | 0                 | 0.00%               |
| hd_hours_per_week       | float64         | 0                 | 0.00%               |
| sports_hours_per_week   | float64         | 0                 | 0.00%               |
| channels_watched        | int64           | 0                 | 0.00%               |
| support_calls_90d       | int64           | 0                 | 0.00%               |
| closure_request_logged  | object          | 0                 | 0.00%               |
| next_cycle_recharge_inr | float64         | 516               | 21.37%              |
| churned                 | int64           | 0                 | 0.00%               |

**Recharges**

| **Column** | **dtype** | **Missing** | **Missing %** |
| ---------------- | --------------- | ----------------- | ------------------- |
| recharge_id      | object          | 0                 | 0.00%               |
| subscriber_id    | object          | 0                 | 0.00%               |
| recharge_date    | object          | 0                 | 0.00%               |
| amount_inr       | object          | 0                 | 0.00%               |
| status           | object          | 0                 | 0.00%               |
| channel          | object          | 0                 | 0.00%               |

| **Raw audit check**         | **Result** |
| --------------------------------- | ---------------- |
| Subscriber exact duplicate rows   | 15               |
| Distinct subscriber IDs           | 2,400            |
| Blank ages                        | 99               |
| Ages outside 10–100              | 6                |
| Blank payment modes               | 116              |
| Distinct raw city spellings       | 22               |
| Distinct raw auto-renew spellings | 6                |
| Blank next-cycle recharge labels  | 516              |
| Recharge exact duplicate rows     | 60               |
| Distinct recharge IDs             | 12,995           |
| Non-numeric recharge amounts      | 366              |
| Distinct raw status spellings     | 6                |
| Orphan recharge rows              | 10               |

### Cleaning rules and results

- Removed 15 exact duplicate subscriber rows, leaving **2,400** subscribers.
- Trimmed city strings, lower-cased for matching, and mapped Bangalore→Bengaluru, Bombay→Mumbai, Madras→Chennai, Calcutta→Kolkata, New Delhi→Delhi and Hyd→Hyderabad. Final city counts:

| **City** | **Subscribers** |
| -------------- | --------------------- |
| Mumbai         | 524                   |
| Delhi          | 436                   |
| Chennai        | 391                   |
| Bengaluru      | 386                   |
| Hyderabad      | 262                   |
| Pune           | 245                   |
| Kolkata        | 156                   |

- Normalised `auto_renew` to Yes/No: No **1,355**, Yes **1,045**.
- Converted ages to numeric and set the **6** nonblank impossible ages outside 10–100 to missing. After duplicate removal and cleaning, age has **104** missing values; payment_mode retains **114** missing values.
- Removed 60 duplicate recharge rows and 10 orphan rows, leaving **12,985**. Lower-cased status, parsed recharge dates and rebuilt **364** non-numeric amounts from the subscriber's `monthly_recharge_inr`.
- Final recharge status counts: success **11,442**, failed **1,422**, pending **121**. Successful recharge total: **INR 6,457,573**.
- Churn labels: **514 / 2,400 (21.42%)**.

No missing values were imputed in Part A. Missing age and payment-mode values are retained so that Part B can learn imputation statistics from training rows only; recharge amount repair is a deterministic reconstruction explicitly required by the brief, not statistical imputation.

The clean `subscribers`, `recharges` and unchanged `plans` tables are written to CSV and SQLite (`data/tataplay.db`). `sql/queries.sql` contains the four required queries.

### SQL results

**Q1 — successful recharge revenue by plan**

| **Plan** | **Revenue (INR)** | **Average (INR)** | **Distinct subscribers** |
| -------------- | ----------------------- | ----------------------- | ------------------------------ |
| Premium        | 1,471,199               | 860.35                  | 339                            |
| Standard       | 1,323,622               | 492.60                  | 559                            |
| Standard HD    | 1,283,471               | 645.61                  | 407                            |
| Premium Plus   | 1,262,398               | 1075.30                 | 228                            |
| Basic          | 782,554                 | 326.61                  | 493                            |
| Basic Lite     | 334,329                 | 224.83                  | 298                            |

A pandas groupby independently confirms Premium revenue of INR 1,471,199 and average recharge INR 860.35.

**Q2 — subscribers with no successful recharge:** **76**, of whom **68** have zero tenure. An INNER JOIN would discard subscriber rows without a matching successful recharge, hiding exactly these customers rather than exposing them for investigation. A LEFT JOIN preserves those subscriber rows, after which the `IS NULL` condition identifies the 76 with no successful recharge.

**Q3 — number of subscribers tied at RANK() = 1 per city**

| **City** | **RANK() = 1 count** |
| -------------- | -------------------------- |
| Bengaluru      | 10                         |
| Chennai        | 10                         |
| Delhi          | 5                          |
| Hyderabad      | 5                          |
| Kolkata        | 2                          |
| Mumbai         | 6                          |
| Pune           | 4                          |

Kolkata's first four rows:

| **Subscriber** | **Total (INR)** | **RANK** | **DENSE_RANK** | **ROW_NUMBER** |
| -------------------- | --------------------- | -------------- | -------------------- | -------------------- |
| TP00074              | 6,876                 | 1              | 1                    | 1                    |
| TP00917              | 6,876                 | 1              | 1                    | 2                    |
| TP00143              | 6,582                 | 3              | 2                    | 3                    |
| TP01355              | 6,582                 | 3              | 2                    | 4                    |

Use `ROW_NUMBER()` when the request is exactly one top spender per city, because it breaks ties deterministically with subscriber_id; `RANK()` and `DENSE_RANK()` deliberately return multiple first-place rows when totals tie.

**Q4 — TP00010 successful recharge running total**

| **Recharge ID** | **Date** | **Amount (INR)** | **Running total (INR)** |
| --------------------- | -------------- | ---------------------- | ----------------------------- |
| R000053               | 2025-10-15     | 498                    | 498                           |
| R000055               | 2025-12-15     | 498                    | 996                           |
| R000056               | 2026-01-15     | 498                    | 1494                          |
| R000057               | 2026-02-15     | 498                    | 1992                          |
| R000058               | 2026-03-13     | 498                    | 2490                          |

The final running total is **INR 2,490** across 5 successful recharges.

## Part B

### Problem framing

| **Question** | **Answer**                                                                                  |
| ------------------ | ------------------------------------------------------------------------------------------------- |
| Decision           | Which subscribers should receive proactive retention outreach before the April 2026 churn window? |
| Who acts?          | Retention desk; prioritise high-risk accounts and tailor an offer or support intervention.        |
| Target             | `churned`                                                                                       |
| Target type        | Binary category (0 = stays, 1 = churns)                                                           |
| Cost of errors     | False negative: INR 1,200 lost revenue; false positive: INR 300 unnecessary offer                 |

Class balance: **514 / 2,400 (21.42%)** churners.

### Features, leakage and split order

The modelling table adds `n_success_6m`, `n_failed_6m` and `recency_days` from recharge rows up to 2026-03-31. `recency_days` is missing for **76** subscribers. There are 12 numeric and 4 categorical predictors.

The split order is: first reserve 20% as test (`test_size=0.2`, stratified, `random_state=42`), then split the remaining 80% into train and validation with 25% of that remainder as validation. Sizes and churners:

| **Set** | **Rows** | **Churners** |
| ------------- | -------------- | ------------------ |
| Train         | 1440           | 308                |
| Validation    | 480            | 103                |
| Test          | 480            | 103                |

Train fits preprocessing and model parameters; validation selects model settings and alert threshold; test is used once after all choices are frozen, for final reporting only.

`closure_request_logged` and `next_cycle_recharge_inr` are excluded because they are recorded during/after the churn outcome. Leakage experiment on validation:

| **Feature set** | **Accuracy** | **Churn F1** |
| --------------------- | ------------------ | ------------------ |
| With leakage          | 0.983              | 0.961              |
| Without leakage       | 0.823              | 0.514              |

The leaky model sees information that would not exist when the retention desk needs to act, so its near-perfect validation score is not deployable. Its apparent performance is an artefact of target leakage, not genuine early-warning ability.

### Preprocessing, baselines and overfitting

The `ColumnTransformer` applies median imputation then scaling to numeric columns, and most-frequent imputation then one-hot encoding to categorical columns. It was fitted only on the training rows; it produces **31** columns, learns median age **38.0**, and learns a tenure scaling mean of **22.65** versus the full-data mean **22.46**. The difference is expected because the scaler sees only the training subset. The transformed training matrix contains no missing values.

| **Model**            | **Train accuracy** | **Validation accuracy** | **Validation precision** | **Validation recall** | **Validation churn F1** |
| -------------------------- | ------------------------ | ----------------------------- | ------------------------------ | --------------------------- | ----------------------------- |
| Majority-class baseline    | —                       | 0.785                         | —                             | —                          | 0.000                         |
| Logistic regression        | —                       | 0.833                         | 0.646                          | 0.495                       | 0.560                         |
| Unrestricted decision tree | 1.000                    | —                            | —                             | —                          | 0.459                         |
| Default random forest      | 1.000                    | —                            | —                             | —                          | 0.514                         |

Decision-tree depth sweep:

| **Max depth** | **Train accuracy** | **Validation churn F1** |
| ------------------- | ------------------------ | ----------------------------- |
| 1                   | 0.815                    | 0.434                         |
| 2                   | 0.832                    | 0.413                         |
| 3                   | 0.839                    | 0.408                         |
| 4                   | 0.844                    | 0.422                         |
| 5                   | 0.863                    | 0.529                         |
| 6                   | 0.883                    | 0.545                         |
| 7                   | 0.908                    | 0.488                         |
| 8                   | 0.924                    | 0.489                         |
| 9                   | 0.944                    | 0.468                         |
| 10                  | 0.957                    | 0.440                         |

The widening train/validation gap at greater depth shows that the tree increasingly memorises training examples instead of generalising. A greedy tree repeatedly selects locally useful splits and can isolate noise or rare patterns, so unconstrained depth overfits.

### Cross-validation tuning

Both searches use `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` and `GridSearchCV(scoring='f1')` on training rows only. The random-forest grid has 3 × 5 × 3 = 45 parameter combinations and 225 fits; logistic regression has 5 combinations and 25 fits.

| **Model** | **Best parameters**                                                     | **Mean CV F1** | **Std F1** | **Combinations** | **Fits** |
| --------------- | ----------------------------------------------------------------------------- | -------------------- | ---------------- | ---------------------- | -------------- |
| logistic        | {'model__C': 100}                                                             | 0.507                | 0.035            | 5                      | 25             |
| random_forest   | {'model__max_depth': 8, 'model__max_features': 5, 'model__n_estimators': 100} | 0.439                | 0.061            | 45                     | 225            |

Logistic regression is more consistent across folds here (standard deviation 0.035 versus 0.061); lower fold-to-fold variation makes its expected performance less sensitive to the particular training fold.

### Cost-based threshold and final test evaluation

The threshold is selected on validation only by minimising `1,200 × false negatives + 300 × false positives` over 0.10–0.90 in 0.05 increments:

| **Model** | **Validation threshold** | **Validation cost** |
| --------------- | ------------------------------ | ------------------------- |
| logistic        | 0.20                           | INR 56,400                |
| random_forest   | 0.30                           | INR 53,100                |

The random forest is selected because it has the lower validation cost. Logistic regression is easier to explain through signed coefficients and has a higher test recall, but the random forest has a lower validation cost under the specified error penalties and its top feature importances still provide a useful operational explanation. The desk can use the random-forest feature rankings to understand why an account is prioritised, while the simpler logistic model remains a useful transparent benchmark. Only after that selection were the held-out test probabilities evaluated:

| **Model** | **TN / FP / FN / TP** | **Precision** | **Recall** | **F1** | **Test cost** |
| --------------- | --------------------------- | ------------------- | ---------------- | ------------ | ------------------- |
| logistic        | 275 / 102 / 17 / 86         | 0.457               | 0.835            | 0.591        | INR 51,000          |
| random_forest   | 310 / 67 / 29 / 74          | 0.525               | 0.718            | 0.607        | INR 54,900          |

Reference policies on the same test set: no action costs **INR 123,600**; offering everybody costs **INR 113,100**. The selected random forest costs 44.4% of no-action cost, with recall 0.718 and F1 0.607, meeting the required test bars. Although logistic regression has stronger test recall and lower realised test cost, model selection follows the validation cost as required. The random forest is the selected model and is persisted to `outputs/churn_model.joblib`.

### Model explanation and persistence

Top random-forest importances (pipeline feature names):

| **Feature**      | **Importance** |
| ---------------------- | -------------------- |
| num__support_calls_90d | 0.196                |
| num__recency_days      | 0.091                |
| num__tenure_months     | 0.082                |
| num__n_success_6m      | 0.079                |
| num__hd_hours_per_week | 0.067                |

Logistic-regression coefficient signs:

| **Feature**      | **Coefficient** |
| ---------------------- | --------------------- |
| num__support_calls_90d | +1.075                |
| num__tenure_months     | -0.511                |
| num__n_failed_6m       | +0.211                |

More support calls and failed recharges increase predicted churn risk; longer tenure decreases it. The fitted preprocessing and model are saved together. Reloading the selected pipeline reproduces the original test probabilities exactly: **True**. `outputs/test_scores.csv` contains 480 rows with `subscriber_id`, `churn_probability` and `churned`.

No imputer, scaler, encoder, cross-validation search or threshold search saw test rows. The test set was held out first, then the train/validation split and all tuning/threshold decisions were completed before the test was used once.

## Part C

### K-means selection and segment profile

K-means uses only tenure, weekly viewing hours and monthly recharge, standardised across all 2,400 clean subscribers. Churn is used only afterwards to describe segments, not to create them. The elbow and business interpretability support **K = 4**.

| **K** | **Inertia** | **Silhouette** |
| ----------- | ----------------- | -------------------- |
| 2           | 3728.2            | 0.420                |
| 3           | 2813.0            | 0.346                |
| 4           | 2193.9            | 0.340                |
| 5           | 1781.1            | 0.343                |
| 6           | 1502.5            | 0.344                |
| 7           | 1292.8            | 0.364                |
| 8           | 1191.4            | 0.336                |

Inertia cannot increase when K increases because adding centroids can only reduce or preserve each point's distance to its nearest centroid, so K=8 naturally has lower inertia than K=2. K=2 has the highest silhouette because the broadest split is easiest to separate, but it collapses useful customer behaviours into only two groups; neither metric alone captures the retention desk's need for actionable segments.

| **ID** | **Segment** | **Size** | **Mean tenure (months)** | **Mean hours/week** | **Mean monthly recharge (INR)** | **Churn rate** | **Recommended action**                               |
| ------------ | ----------------- | -------------- | ------------------------------ | ------------------------- | ------------------------------------- | -------------------- | ---------------------------------------------------------- |
| 0            | Value             | 593            | 23.6                           | 7.1                       | 301.33                                | 22.43%               | Offer low-cost plan optimisation                           |
| 1            | Engaged           | 543            | 34.2                           | 22.2                      | 559.73                                | 13.44%               | Promote content bundles aligned with viewing               |
| 2            | Premium           | 480            | 34.2                           | 33.6                      | 942.32                                | 10.62%               | Protect premium experience; offer relevant add-ons         |
| 3            | At-Risk           | 784            | 6.3                            | 14.9                      | 513.51                                | 32.78%               | Proactive retention outreach and recharge-friction support |

Cluster IDs are arbitrary labels, not ranks. The segment with the highest churn rate is named At-Risk; the highest mean monthly recharge segment is Premium.

New subscriber assignment (tenure 4 months, 15 weekly hours, INR 520 monthly): scaled Euclidean distances by cluster ID are **1.708, 2.094, 3.117, 0.153**. The nearest is cluster **3 (At-Risk)**, and `kmeans.predict` agrees (cluster 3).

### Correlation and spend prediction

Correlation matrix for the four viewing features:

| **Feature**      | **hours_watched_per_week** | **hd_hours_per_week** | **sports_hours_per_week** | **channels_watched** |
| ---------------------- | -------------------------------- | --------------------------- | ------------------------------- | -------------------------- |
| hours_watched_per_week | 1.000                            | 0.979                       | 0.947                           | 0.938                      |
| hd_hours_per_week      | 0.979                            | 1.000                       | 0.927                           | 0.921                      |
| sports_hours_per_week  | 0.947                            | 0.927                       | 1.000                           | 0.891                      |
| channels_watched       | 0.938                            | 0.921                       | 0.891                           | 1.000                      |

Pairs with absolute correlation above 0.9: hours–HD **0.979**, hours–sports **0.947**, hours–channels **0.938**, HD–sports **0.927**, and HD–channels **0.921**. Sports–channels is 0.891 and is below the threshold.

The spend model uses only non-churned subscribers and the seven required features. The split is **1,508 train / 378 test**. Ridge and Lasso alpha values were tuned with five-fold CV on the training rows.

| **Model** | **Best alpha** | **Mean CV R²** | **Test R²** | **MAE (INR)** | **RMSE (INR)** |
| --------------- | -------------------- | --------------------- | ------------------ | ------------------- | -------------------- |
| linear          | —                   | 0.9941                | 0.9930             | 20.16               | 24.70                |
| ridge           | 0.01                 | 0.9941                | 0.9930             | 20.16               | 24.70                |
| lasso           | 0.1                  | 0.9941                | 0.9930             | 20.11               | 24.65                |

Lasso's exactly-zero coefficients are **tenure_months, hours_watched_per_week, channels_watched**. With highly correlated predictors, several columns carry overlapping information; Lasso can retain one representative and shrink redundant coefficients to zero without losing much predictive accuracy.

Coefficient stability uses five 80% subsamples and population standard deviation (`np.std`, `ddof=0`) of the standardised `hours_watched_per_week` coefficient:

| **Model** | **Five coefficients**       | **Population SD** |
| --------------- | --------------------------------- | ----------------------- |
| linear          | -5.73, -1.71, -4.84, -7.74, -4.73 | 1.945                   |
| ridge_alpha_30  | 13.26, 15.55, 13.58, 13.02, 13.79 | 0.896                   |

Ridge's coefficient variation is much smaller, showing that regularisation stabilises individual coefficients when predictors are strongly correlated; an individual unregularised coefficient should therefore not be over-interpreted.

### Revenue at risk

For each of the 480 Part B test subscribers, the best Ridge pipeline predicts next-cycle recharge, including subscribers who churned. Revenue at risk is `churn_probability × predicted_next_recharge_inr`. The top 15 have combined expected revenue at risk of **INR 8,716.53**, and **12 / 15** actually churned. The priority list is built from test subscribers whose outcomes are already known, so it illustrates the ranking method and is not a forward-looking evaluation. It is saved to `outputs/retention_priority.csv` with the required six columns.

## Part D

Part D uses the supplied offline mock through `llm_client.py`, LangChain prompt templates and parsers, Pydantic v2 and a ChromaDB cosine collection. `part_d.py` creates the classifier, extraction and RAG outputs; `copilot.py` produces the 15 validated retention briefs. The generated Part D metrics are inserted below by `update_readme.py` after a successful run.

### Client demonstrations

| **Check**                                  | **Observed result** |
| ------------------------------------------------ | ------------------------- |
| Missing Authorization                            | 401                       |
| Wrong URL                                        | 404                       |
| max_tokens=3                                     | length                    |
| No history                                       | I do not know your name.  |
| Same request with history                        | Your name is Asha.        |
| 30 calls: recovered rate limits / physical posts | 3 / 33                    |

Pydantic coerces urgency string `3` to integer **3**. `urgency=9` and `category='refund'` raise field validation errors: `[{"field": "urgency", "error_type": "less_than_equal"}]` and `[{"field": "category", "error_type": "literal_error"}]`.

### Ticket classification

| **Prompt** | **Valid labels** | **Correct** | **Accuracy** | **Recovered 429s** |
| ---------------- | ---------------------- | ----------------- | ------------------ | ------------------------ |
| weak             | 0                      | 0                 | 0.00%              | 18                       |
| zero_shot        | 146                    | 123               | 84.25%             | 18                       |
| few_shot         | 146                    | 142               | 97.26%             | 18                       |

Few-shot errors (the readable ticket IDs and predicted/true labels) are: T0041 (billing vs plan_change), T0094 (signal vs cancellation), T0110 (cancellation vs signal), T0136 (cancellation vs signal). The remaining errors contain multiple intents; a production classifier should identify the primary requested action first or represent multiple intents explicitly. Chain-of-thought is justified only when multi-step reasoning must be audited; these short fixed-label tickets are better handled with concise prompts and examples.

### Validated ticket extraction

| **Measure**                | **Result**                                                                                       |
| -------------------------------- | ------------------------------------------------------------------------------------------------------ |
| Outcome counts                   | {"valid": 146, "manual_review": 4}                                                                     |
| First-attempt failure categories | {"invalid_json_chatty": 12, "urgency_out_of_range": 6, "invalid_category": 10, "unreadable_ticket": 4} |
| Valid records                    | 146                                                                                                    |
| wants_refund = true              | 31                                                                                                     |
| Null subscriber_id               | 75                                                                                                     |
| Recovered rate limits            | 23                                                                                                     |

Tickets still marked `manual_review` are routed to a human support queue with the original text and validation errors; they are not silently discarded or answered with an invented record.

### Policy retrieval and grounded answering

| **Retrieval measure**                | **Result**            |
| ------------------------------------------ | --------------------------- |
| 400/60 chunk count                         | 13                          |
| Answerable top-3 file hits                 | 12 / 12                     |
| Answerable top-1 file hits                 | 12 / 12                     |
| Answerable top-1 similarity min–max–mean | 0.334 – 0.637 – 0.514     |
| Unanswerable top-1 similarities            | 0.069, -0.047, 0.295, 0.304 |
| Keyword baseline top-1 correct             | 9 / 12                      |
| Rejection threshold                        | 0.319                       |
| Similarity gap between ranges              | 0.030                       |
| Grounded answers containing expected fact  | 12 / 12                     |
| Unanswerable questions receiving fallback  | 4 / 4                       |
| Unique ungrounded answers                  | 1                           |
| 200/30 chunk count                         | 25                          |
| 200/30 top-3 fact coverage                 | 11 / 12                     |
| 200/30 grounded expected facts             | 8 / 12                      |

Meaning-based retrieval helps because it can match paraphrases and related concepts even when the question and policy use different words; keyword overlap misses such matches and can be distracted by generic shared terms. The unanswerable question closest to the rejection boundary was **Is international roaming available on the mobile app?** (top-1 similarity 0.304). The observed gap between the weakest answerable and strongest unanswerable question is only 0.030, so the threshold has limited safety margin and needs more labelled calibration examples. Smaller chunks can split a policy sentence or separate a condition from its action, reducing the chance that all evidence appears together in the top three results.

### Retention briefs

| **Measure**           | **Result**                                                                                                                                                                                                                                         |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Valid and consistent briefs | 15                                                                                                                                                                                                                                                       |
| Risk-band counts            | {"High": 14, "Medium": 1}                                                                                                                                                                                                                                |
| Recommended-action counts   | {"Offer a free add-on pack for 2 months.": 6, "Offer an auto-renew setup call and a 7 day grace extension.": 4, "Offer a loyalty discount of 15 percent on the next three recharges.": 4, "No retention offer applies; keep monitoring the account.": 1} |

All 15 briefs are validated against the Pydantic schema, checked against the source subscriber and risk band, and required to cite `retention_offers_policy.txt`.

The retention-offer policy rule implemented by `copilot.py` is:

| **Subscriber condition**                                 | **Required action**                        |
| -------------------------------------------------------------- | ------------------------------------------------ |
| High risk and at least 3 failed recharges in the last 6 months | Auto-renew setup call and 7-day grace extension  |
| Otherwise High risk and tenure of at least 12 months           | 15% loyalty discount on the next three recharges |
| Otherwise High risk                                            | Free add-on pack for 2 months                    |
| Medium or Low risk                                             | No retention offer applies                       |

`copilot.py` reads subscriber and recharge features from SQLite, re-scores all 15 priority subscribers with the saved model, checks the saved probability to four decimals, retrieves policy chunks and validates each retention brief. Expected Part D output paths after a successful run are `outputs/ticket_classification.csv`, `outputs/ticket_extractions.jsonl`, `outputs/rag_eval.csv` and `outputs/retention_briefs.jsonl`. After both Part D scripts complete, `update_readme.py` replaces the marked results block with the measured classifier, extraction, retrieval and brief metrics.

## Limitations and Next Steps

1. The data and policies are synthetic and do not represent actual Tata Play customers or operations.
2. The churn target covers one month after one observation window; performance across later months and changing customer behaviour is untested.
3. The offline mock LLM is deterministic and does not represent the reasoning, variability or failure modes of a production model.
4. The revenue-at-risk priority list is built from held-out test subscribers whose churn outcomes are already known; it demonstrates the calculation rather than a prospective deployment.
5. The RAG rejection threshold separates a narrow similarity gap; embedding/library changes may move borderline questions, so the threshold needs calibration on more labelled questions.
6. The model has not been deployed or monitored; a production system would need privacy/security controls, human review, drift monitoring, policy versioning and operational feedback loops.
