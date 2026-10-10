# Model & Data Guide

> **IMPORTANT NOTICE:** All data generated and processed by this application is **100% synthetic**. There are no real customers, no real transactions, and no real bank partnerships. The data is generated solely for demonstration and evaluation of the AI credit scrutiny platform.

## Synthetic Dataset Generation

The dataset is generated via `backend/app/ml/synth.py`. It simulates a realistic population of ~3,000 Pakistani SMEs and their 24-month transaction histories.

### Generation Assumptions & Methodology

1. **Latent Business Health:** A continuous latent variable $H \sim \mathcal{N}(0, 1)$ bounded in $[-3, 3]$ dictates the underlying financial strength of the business. It causally drives revenue levels, digital payment adoption, and utility bill punctuality.
2. **Demographics & Fairness:**
   - **Population:** Spans 8 sectors (e.g., retail, logistics, agri-processing) across major Pakistani cities and provinces.
   - **Gender & Inclusion:** ~20% of businesses are women-led. To reflect historic financial exclusion, women-led businesses have a higher probability (~50%) of having a "thin file" (no bureau record or <12 months history) compared to others (~31.25%). Overall thin-file share is ~35%.
   - **Fairness Constraint:** Owner gender and age band have **zero causal weight** in determining the true default outcome. Any correlation is due to underlying systemic factors (like the thin-file overlap) rather than a causal penalty.
3. **Transactions:**
   - 24 months of daily transactions (inflows and outflows) are simulated per business.
   - Includes sector-specific seasonality (e.g., Eid and wedding peaks for garments, harvest cycles for agri-processing).
   - Mathematical reconciliation: Every running balance strictly equals the previous balance plus/minus the transaction amount. Balances never drop below zero.
4. **Fraud & Integrity:**
   - 3-5% of the population is injected with specific fraud patterns: Window Dressing, Circular Flows, and Tampered Balances.
   - **Crucial Rule:** The `is_synthetic_fraud` flag is kept strictly in the `loan_evaluations` table. It is **never** exposed in the standard `businesses` or feature tables to prevent label leakage.
5. **Default Outcome:**
   - The 12-month default outcome is generated using a logistic function based on the latent health variable, the presence of injected fraud, and random noise.
   - The overall target default rate is 8–12%, calibrated so a good ML model achieves an AUC of 0.75–0.85 (avoiding perfect separability).

### Demo Personas

The dataset includes three fixed personas with stable IDs (1, 2, 3) for reproducible demos:
1. **Ayesha's Boutique (ID 1):** Thin file, women-led, strong steady digital cash flow. Likely rejected by traditional rules but fundamentally healthy (latent health > 0, default = 0).
2. **Karim Traders (ID 2):** Thick file, borderline latent health.
3. **FastFix Logistics (ID 3):** Clean-looking profile but with injected window dressing prior to the application date. Designed to trigger the integrity/fraud checks (default = 1).

## Data Dictionary

The generated SQLite database (`sme_credit.db`) and CSV exports (`/data`) follow this schema:

### `businesses`
- `id`: Primary key
- `name`: Business name (e.g., "Ayesha's Boutique", "Khan Traders 4")
- `sector`: Industry sector
- `city`, `province`: Location
- `legal_form`: Sole proprietorship, partnership, etc.
- `employees`: Number of employees (1-60)
- `owner_gender`, `owner_age_band`: Demographics (causally decoupled from default)
- `is_women_led`: Boolean flag
- `is_synthetic`: Always `1` (True) for this dataset
- `has_ntn`: National Tax Number registration status
- `collateral_value_pkr`: Self-declared collateral value in PKR

### `applications`
- `id`, `business_id`: Keys
- `product`: e.g., Working Capital, Term Loan
- `amount_requested_pkr`: Loan amount requested
- `tenor_months`: Duration of loan
- `purpose`: Stated purpose

### `transactions`
- `id`, `business_id`: Keys
- `date`: Transaction date
- `amount_pkr`: Transaction amount
- `direction`: Inflow or Outflow
- `channel`: Bank, Wallet, Instant Transfer, Cash Deposit
- `category`: e.g., customer_payment, payroll, utility_bill
- `counterparty_hash`: SHA-256 hash of the counterparty
- `balance_after`: Mathematically reconciled running balance

### `bill_payments` & `bureau_records`
- Track utility punctuality and credit bureau scores (if not thin file).

### `loan_evaluations` (Holdout / Evaluation Table)
- **Warning:** Features in this table must **not** be used during model training.
- `default_12m`: The target label (1 = default, 0 = non-default).
- `latent_business_health`: The true underlying strength used during generation.
- `is_synthetic_fraud`: Flag indicating injected fraud.
- `fraud_pattern`: The specific pattern injected (e.g., "window_dressing").
- `is_thin_file`: Flag indicating lack of sufficient bureau history.

## Feature Engineering

_To be documented when the ML pipeline is built._

## Model Architecture

_To be documented when the default-risk model is trained._

## Evaluation

_To be documented when holdout-set metrics are computed._
