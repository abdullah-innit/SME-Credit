# Illustrative Knowledge Base — How the Score Works
<!-- ILLUSTRATIVE ONLY: For demonstration and simulation purposes in SME Credit. Not official bank or regulatory policy text. -->

## Chunk KB-SCOR-001: Cash-Flow Scoring Architecture (300–900 Scale)
- **Chunk ID:** KB-SCOR-001
- **Category:** how_score_works
- **Title:** Cash-Flow Scoring Architecture (300 to 900 Scorecard)
- **Tags:** score, scorecard, pd, cash-flow, algorithm, machine learning, 300-900

### How the Score is Calculated
The SME Credit score translates an enterprise's verified cash-flow dynamics into an intuitive 300 to 900 credit scale:
- The core engine computes a calibrated 12-month **Probability of Default (PD)** using regularized logistic and gradient-boosted models.
- The PD is mapped monotonically to a 300–900 scorecard where 300 represents extreme default probability and 900 represents minimal credit risk.
- **Key Input Pillars:**
  1. *Cash Velocity & Stability:* Daily transaction turnover, inflow variance, month-on-month revenue momentum.
  2. *Buffer Discipline:* Minimum monthly balances, days with zero balance, overdraft frequencies.
  3. *Bill & Obligation Punctuality:* Historical utility bill payment timeliness and bounce penalties.
  4. *Digital Footprint:* Share of transactions conducted through digital banking and Raast rails versus physical cash deposits.

*Urdu Summary:* اسکور 300 سے 900 کے پیمانے پر اصل کیش فلو، آمدن کے استحکام، بلوں کی بروقت ادائیگی اور ڈیجیٹل لین دین کی بنیاد پر تیار کیا جاتا ہے۔

---

## Chunk KB-SCOR-002: Risk Tiers A Through E and Approval Thresholds
- **Chunk ID:** KB-SCOR-002
- **Category:** how_score_works
- **Title:** Risk Tiers (A to E) and Underwriting Decision Thresholds
- **Tags:** risk tier, tiers, approve, refer, decline, cutoff, probability of default

### Risk Tier Definitions
Every score falls into one of five standardized risk tiers:
- **Tier A (Score 750–900 | PD < 3.0%):** Strongest financial health. Automatic eligibility for fast-track **APPROVE** decision with prime rate bands.
- **Tier B (Score 680–749 | PD 3.0%–7.5%):** Healthy operating SME. Standard **APPROVE** decision with regular working capital terms.
- **Tier C (Score 620–679 | PD 7.5%–14.0%):** Moderate risk. System recommendation is **REFER** for credit officer scrutiny; often approved with condition covenants or collateral/guarantor mitigants.
- **Tier D (Score 550–619 | PD 14.0%–25.0%):** Elevated risk. System recommendation is **REFER** or **DECLINE** depending on policy rules; requires senior underwriter review.
- **Tier E (Score 300–549 | PD > 25.0%):** High delinquency risk. Standard outcome is **DECLINE**; adverse action reasons and custom Path to Approval roadmap are generated.

*Urdu Summary:* اسکور کو پانچ درجات A تا E میں تقسیم کیا جاتا ہے۔ 680 سے اوپر عام طور پر منظور، 620 تا 679 ریفر (افسر کا جائزہ) اور 550 سے نیچے مسترد کیا جاتا ہے۔

---

## Chunk KB-SCOR-003: Explainability and Adverse-Action Reasons
- **Chunk ID:** KB-SCOR-003
- **Category:** how_score_works
- **Title:** Explainability Waterfall and Adverse-Action Notices
- **Tags:** explainability, shap, reasons, adverse action, transparency, fairness

### Reconciled Feature Contribution
SME Credit operates under strict algorithmic transparency:
- Every score calculation generates an exact feature contribution waterfall (local log-odds decomposition).
- The applicant and credit officer see the top factors that **improved** the score and the top factors that **lowered** the score.
- When an application is referred or declined, the platform issues plain-language bilingual **Adverse-Action Notices** detailing the specific financial metrics (e.g., elevated cash-flow volatility, insufficient digital ratio, or bill payment delays) that affected the outcome.

*Urdu Summary:* ہر اسکور کے ساتھ مثبت اور منفی وجوہات کا شفاف چارٹ دیا جاتا ہے تاکہ کسٹمر کو معلوم ہو کہ کس چیز نے اسکور پر اثر ڈالا۔
