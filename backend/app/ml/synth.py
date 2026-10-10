"""Synthetic SME population generator for credit assessment and fraud detection.

Generates realistic, honestly labelled synthetic data for ~3,000 Pakistani SMEs
over 24 months of daily transaction history, with latent business health driving
revenue, volatility, seasonality, utility punctuality, and digital adoption.

CLI usage:
    python -m app.ml.synth --n 3000 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sqlite3
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import expit

from app import enums
from app.db.base import Base
from app.db.session import make_engine

# ---------------------------------------------------------------------------
# Constants & Reference Distributions
# ---------------------------------------------------------------------------

SECTORS = [
    "retail",
    "garments and textiles",
    "food and beverage",
    "agri-processing",
    "IT and freelance services",
    "logistics",
    "light manufacturing",
    "clinics",
]
SECTOR_WEIGHTS = [0.20, 0.18, 0.15, 0.12, 0.10, 0.09, 0.10, 0.06]

CITIES_PROVINCES = [
    ("Karachi", "Sindh", 0.28),
    ("Lahore", "Punjab", 0.24),
    ("Faisalabad", "Punjab", 0.10),
    ("Rawalpindi", "Punjab", 0.08),
    ("Gujranwala", "Punjab", 0.06),
    ("Multan", "Punjab", 0.05),
    ("Peshawar", "Khyber Pakhtunkhwa", 0.05),
    ("Quetta", "Balochistan", 0.03),
    ("Islamabad", "Islamabad Capital Territory", 0.04),
    ("Sialkot", "Punjab", 0.04),
    ("Hyderabad", "Sindh", 0.03),
]

LEGAL_FORMS = [
    (enums.LegalForm.SOLE_PROPRIETORSHIP, 0.65),
    (enums.LegalForm.PARTNERSHIP, 0.20),
    (enums.LegalForm.PRIVATE_LIMITED, 0.13),
    (enums.LegalForm.OTHER, 0.02),
]

AGE_BANDS = [
    enums.AgeBand.AGE_25_34,
    enums.AgeBand.AGE_35_44,
    enums.AgeBand.AGE_45_54,
    enums.AgeBand.AGE_55_PLUS,
    enums.AgeBand.UNDER_25,
]
AGE_WEIGHTS = [0.35, 0.32, 0.20, 0.08, 0.05]

START_DATE = date(2024, 10, 1)
END_DATE = date(2026, 9, 30)
TOTAL_DAYS = (END_DATE - START_DATE).days + 1  # 730 days

# Precomputed deterministic counterparty hashes
_HASH_POOL = [hashlib.sha256(f"counterparty_seed_{i}".encode()).hexdigest() for i in range(2000)]

# ---------------------------------------------------------------------------
# Business Persona Definitions (Fixed IDs 1, 2, 3)
# ---------------------------------------------------------------------------

PERSONA_1_AYESHA = {
    "id": 1,
    "name": "Ayesha's Boutique",
    "sector": "garments and textiles",
    "city": "Lahore",
    "province": "Punjab",
    "legal_form": enums.LegalForm.SOLE_PROPRIETORSHIP,
    "incorporated_on": date(2023, 3, 15),
    "employees": 6,
    "owner_gender": enums.Gender.FEMALE,
    "owner_age_band": enums.AgeBand.AGE_25_34,
    "is_women_led": True,
    "has_ntn": True,
    "collateral_value_pkr": 0,
    "latent_health": 1.85,
    "is_thin_file": True,
    "is_synthetic_fraud": 0,
    "fraud_pattern": None,
    "default_12m": 0,
    "product": enums.Product.WORKING_CAPITAL,
    "amount_requested_pkr": 750000,
    "tenor_months": 12,
    "purpose": "Inventory expansion for Eid collection",
}

PERSONA_2_KARIM = {
    "id": 2,
    "name": "Karim Traders",
    "sector": "retail",
    "city": "Karachi",
    "province": "Sindh",
    "legal_form": enums.LegalForm.PARTNERSHIP,
    "incorporated_on": date(2018, 7, 10),
    "employees": 14,
    "owner_gender": enums.Gender.MALE,
    "owner_age_band": enums.AgeBand.AGE_45_54,
    "is_women_led": False,
    "has_ntn": True,
    "collateral_value_pkr": 2500000,
    "latent_health": 0.25,
    "is_thin_file": False,
    "is_synthetic_fraud": 0,
    "fraud_pattern": None,
    "default_12m": 0,
    "product": enums.Product.WORKING_CAPITAL,
    "amount_requested_pkr": 2000000,
    "tenor_months": 12,
    "purpose": "Bulk seasonal FMCG procurement",
}

PERSONA_3_FASTFIX = {
    "id": 3,
    "name": "FastFix Logistics",
    "sector": "logistics",
    "city": "Rawalpindi",
    "province": "Punjab",
    "legal_form": enums.LegalForm.PRIVATE_LIMITED,
    "incorporated_on": date(2020, 11, 20),
    "employees": 22,
    "owner_gender": enums.Gender.MALE,
    "owner_age_band": enums.AgeBand.AGE_35_44,
    "is_women_led": False,
    "has_ntn": True,
    "collateral_value_pkr": 5000000,
    "latent_health": -1.40,
    "is_thin_file": False,
    "is_synthetic_fraud": 1,
    "fraud_pattern": "window_dressing",
    "default_12m": 1,
    "product": enums.Product.TERM_LOAN,
    "amount_requested_pkr": 3500000,
    "tenor_months": 24,
    "purpose": "Fleet expansion and maintenance depot",
}


def get_sector_seasonality(sector: str, month: int) -> float:
    """Return seasonal multiplier for sector in given calendar month (1-12)."""
    if sector in ("retail", "garments and textiles"):
        # Eid ul-Fitr (March/April), Eid ul-Adha (June), Winter wedding season (Nov-Jan)
        if month in (3, 4):
            return 1.55
        if month == 6:
            return 1.45
        if month in (11, 12, 1):
            return 1.35
        return 0.90
    if sector == "agri-processing":
        # Rabi harvest (April/May) & Kharif harvest (October/November)
        if month in (4, 5):
            return 1.65
        if month in (10, 11):
            return 1.60
        if month in (7, 8):
            return 0.70
        return 0.95
    if sector == "food and beverage":
        # Ramzan dining surge (March/April) and summer beverages (May-July)
        if month in (3, 4):
            return 1.40
        if month in (5, 6, 7):
            return 1.30
        return 0.90
    if sector == "logistics":
        # Trade & harvest transit quarters
        if month in (10, 11, 12):
            return 1.30
        if month in (3, 4):
            return 1.25
        return 0.95
    # Clinics, IT, light manufacturing: steady baseline
    return 1.00


# ---------------------------------------------------------------------------
# Core Synthesis Class
# ---------------------------------------------------------------------------


class SyntheticDataGenerator:
    """Generates synthetic SME credit dataset."""

    def __init__(self, n: int = 3000, seed: int = 42):
        self.n = max(n, 3)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate_population(self) -> dict[str, Any]:
        """Generate full population records and evaluation outcomes."""
        n = self.n

        # 1. Latent Health Variable H ~ N(0, 1), clipped to [-3, 3]
        latent_health = self.rng.normal(0, 1, n)
        latent_health = np.clip(latent_health, -3.0, 3.0)

        # 2. Demographics & Fairness Design
        # Women-led share: ~20%
        is_women_led = np.zeros(n, dtype=bool)
        is_women_led[3:] = self.rng.uniform(0, 1, n - 3) < 0.20

        # Owner Gender: deterministic from women-led (female) vs male/other
        owner_gender = []
        for i in range(n):
            if is_women_led[i]:
                owner_gender.append(enums.Gender.FEMALE)
            else:
                rand = self.rng.uniform(0, 1)
                if rand < 0.96:
                    owner_gender.append(enums.Gender.MALE)
                elif rand < 0.98:
                    owner_gender.append(enums.Gender.OTHER)
                else:
                    owner_gender.append(enums.Gender.UNDISCLOSED)

        # Thin-file share: ~35% overall
        # Documented historic exclusion assumption: women ~50%, men ~31.25%
        is_thin_file = np.zeros(n, dtype=bool)
        for i in range(3, n):
            prob = 0.50 if is_women_led[i] else 0.3125
            is_thin_file[i] = self.rng.uniform(0, 1) < prob

        # NTN registration: ~60%
        has_ntn = np.zeros(n, dtype=bool)
        has_ntn[3:] = self.rng.uniform(0, 1, n - 3) < 0.60

        # Sectors
        chosen_sectors = self.rng.choice(SECTORS, size=n - 3, p=SECTOR_WEIGHTS).tolist()
        sectors = [
            PERSONA_1_AYESHA["sector"],
            PERSONA_2_KARIM["sector"],
            PERSONA_3_FASTFIX["sector"],
        ] + chosen_sectors

        # Cities & Provinces
        city_names = [c[0] for c in CITIES_PROVINCES]
        city_probs = np.array([c[2] for c in CITIES_PROVINCES])
        city_probs /= city_probs.sum()
        city_map = {c[0]: c[1] for c in CITIES_PROVINCES}

        chosen_cities = self.rng.choice(city_names, size=n - 3, p=city_probs).tolist()
        cities = [
            PERSONA_1_AYESHA["city"],
            PERSONA_2_KARIM["city"],
            PERSONA_3_FASTFIX["city"],
        ] + chosen_cities
        provinces = [city_map[c] for c in cities]

        # Legal Forms
        lf_forms = [lf[0] for lf in LEGAL_FORMS]
        lf_probs = [lf[1] for lf in LEGAL_FORMS]
        chosen_lf = self.rng.choice(lf_forms, size=n - 3, p=lf_probs).tolist()
        legal_forms = [
            PERSONA_1_AYESHA["legal_form"],
            PERSONA_2_KARIM["legal_form"],
            PERSONA_3_FASTFIX["legal_form"],
        ] + chosen_lf

        # Age Bands
        chosen_age = self.rng.choice(AGE_BANDS, size=n - 3, p=AGE_WEIGHTS).tolist()
        age_bands = [
            PERSONA_1_AYESHA["owner_age_band"],
            PERSONA_2_KARIM["owner_age_band"],
            PERSONA_3_FASTFIX["owner_age_band"],
        ] + chosen_age

        # Employees: 1-60 (skewed distribution)
        raw_emp = self.rng.exponential(scale=7.0, size=n - 3) + 1.0
        emp_list = [
            PERSONA_1_AYESHA["employees"],
            PERSONA_2_KARIM["employees"],
            PERSONA_3_FASTFIX["employees"],
        ] + np.clip(raw_emp.astype(int), 1, 60).tolist()

        # Injected Fraud: 3-5%
        is_synthetic_fraud = np.zeros(n, dtype=int)
        fraud_patterns: list[str | None] = [None] * n
        is_synthetic_fraud[2] = 1
        fraud_patterns[2] = "window_dressing"

        fraud_types = ["window_dressing", "circular_flows", "tampered_balances"]
        fraud_probs = [0.40, 0.35, 0.25]
        for i in range(3, n):
            if self.rng.uniform(0, 1) < 0.039:
                is_synthetic_fraud[i] = 1
                fraud_patterns[i] = self.rng.choice(fraud_types, p=fraud_probs)

        # Default Outcome: 12-month default rate 8-12%, AUC 0.75-0.85
        # Causal equation: gender and age have strict 0.0 causal weight
        noise = self.rng.normal(0, 0.85, n)
        z = -3.08 - 1.55 * latent_health + 1.25 * is_synthetic_fraud.astype(float) + noise
        p_default = expit(z)
        default_12m = (p_default > self.rng.uniform(0, 1, n)).astype(int)

        # Apply Fixed Personas
        latent_health[0] = PERSONA_1_AYESHA["latent_health"]
        is_women_led[0] = True
        owner_gender[0] = PERSONA_1_AYESHA["owner_gender"]
        is_thin_file[0] = True
        has_ntn[0] = True
        is_synthetic_fraud[0] = 0
        fraud_patterns[0] = None
        default_12m[0] = 0

        latent_health[1] = PERSONA_2_KARIM["latent_health"]
        is_women_led[1] = False
        owner_gender[1] = PERSONA_2_KARIM["owner_gender"]
        is_thin_file[1] = False
        has_ntn[1] = True
        is_synthetic_fraud[1] = 0
        fraud_patterns[1] = None
        default_12m[1] = 0

        latent_health[2] = PERSONA_3_FASTFIX["latent_health"]
        is_women_led[2] = False
        owner_gender[2] = PERSONA_3_FASTFIX["owner_gender"]
        is_thin_file[2] = False
        has_ntn[2] = True
        is_synthetic_fraud[2] = 1
        fraud_patterns[2] = "window_dressing"
        default_12m[2] = 1

        # Business names
        business_names = [
            PERSONA_1_AYESHA["name"],
            PERSONA_2_KARIM["name"],
            PERSONA_3_FASTFIX["name"],
        ]
        surnames = [
            "Khan",
            "Ahmed",
            "Malik",
            "Chaudhry",
            "Bhatti",
            "Rehman",
            "Siddiqui",
            "Shah",
            "Qureshi",
            "Butt",
            "Tariq",
            "Iqbal",
            "Zubair",
            "Farooq",
            "Hussain",
            "Mirza",
            "Ansari",
            "Dar",
            "Mughal",
            "Raza",
        ]
        biz_suffixes = [
            "Traders",
            "Enterprises",
            "Mills",
            "Works",
            "Services",
            "Logistics",
            "Solutions",
            "Mart",
            "Pharmacy",
            "Foods",
            "Creations",
            "Fabrics",
            "Engineering",
            "Supplies",
            "Agri",
            "Motors",
        ]
        for i in range(3, n):
            s = surnames[i % len(surnames)]
            suff = biz_suffixes[(i * 7 + 3) % len(biz_suffixes)]
            business_names.append(f"{s} {suff} {i + 1}")

        # Collateral value (PKR)
        collateral_values = np.zeros(n, dtype=int)
        collateral_values[0] = 0
        collateral_values[1] = PERSONA_2_KARIM["collateral_value_pkr"]
        collateral_values[2] = PERSONA_3_FASTFIX["collateral_value_pkr"]
        for i in range(3, n):
            if is_thin_file[i]:
                # Thin file businesses predominantly have 0 or modest collateral
                collateral_values[i] = (
                    0
                    if self.rng.uniform(0, 1) < 0.85
                    else int(self.rng.choice([250000, 500000, 750000]))
                )
            else:
                collateral_values[i] = int(
                    self.rng.choice([1000000, 1500000, 2000000, 3000000, 5000000, 8000000])
                )

        return {
            "n": n,
            "latent_health": latent_health,
            "is_women_led": is_women_led,
            "owner_gender": owner_gender,
            "is_thin_file": is_thin_file,
            "has_ntn": has_ntn,
            "sectors": sectors,
            "cities": cities,
            "provinces": provinces,
            "legal_forms": legal_forms,
            "age_bands": age_bands,
            "employees": emp_list,
            "business_names": business_names,
            "collateral_values": collateral_values,
            "is_synthetic_fraud": is_synthetic_fraud,
            "fraud_patterns": fraud_patterns,
            "default_12m": default_12m,
        }

    def generate_all(
        self,
        db_path: Path | str = "backend/sme_credit.db",
        data_dir: Path | str = "data",
        samples_dir: Path | str = "samples",
        quiet: bool = False,
    ) -> dict[str, Any]:
        """Execute full generation and write outputs."""
        pop = self.generate_population()
        n = pop["n"]

        # Ensure directories exist
        data_p = Path(data_dir).resolve()
        data_p.mkdir(parents=True, exist_ok=True)
        samples_p = Path(samples_dir).resolve()
        samples_p.mkdir(parents=True, exist_ok=True)
        db_p = Path(db_path).resolve()
        db_p.parent.mkdir(parents=True, exist_ok=True)

        if not quiet:
            print(f"Generating synthetic dataset: N={n}, seed={self.seed}")

        # Initialize SQLite database
        engine = make_engine(f"sqlite:///{db_p}")
        Base.metadata.create_all(engine)

        conn = sqlite3.connect(str(db_p))
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA synchronous = OFF")
        conn.execute("PRAGMA journal_mode = MEMORY")

        # Create evaluation table
        conn.execute(
            """CREATE TABLE IF NOT EXISTS loan_evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                business_id INTEGER NOT NULL,
                application_id INTEGER NOT NULL,
                default_12m INTEGER NOT NULL,
                latent_business_health REAL NOT NULL,
                is_synthetic_fraud INTEGER NOT NULL,
                fraud_pattern TEXT,
                is_thin_file INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (business_id) REFERENCES businesses (id),
                FOREIGN KEY (application_id) REFERENCES applications (id)
            )"""
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS ix_loan_evaluations_business_id "
            "ON loan_evaluations (business_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS ix_loan_evaluations_application_id "
            "ON loan_evaluations (application_id)"
        )

        now_iso = datetime.now(UTC).isoformat().replace("+00:00", "Z")

        # -------------------------------------------------------------------
        # 1. Users, Businesses, Consents, Applications, Documents
        # -------------------------------------------------------------------
        users_rows = []
        businesses_rows = []
        consents_rows = []
        apps_rows = []
        docs_rows = []
        eval_rows = []
        bureau_rows = []
        bill_rows = []

        products = [
            enums.Product.WORKING_CAPITAL,
            enums.Product.TERM_LOAN,
            enums.Product.INVOICE_FINANCING,
        ]

        doc_id_counter = 1
        bill_id_counter = 1
        bureau_id_counter = 1

        for i in range(n):
            biz_id = i + 1
            user_id = i + 1
            app_id = i + 1

            # User
            users_rows.append((user_id, now_iso, now_iso, enums.UserRole.APPLICANT))

            # Business
            inc_date = date(2020 + (i % 4), (i % 12) + 1, (i % 25) + 1)
            if i == 0:
                inc_date = PERSONA_1_AYESHA["incorporated_on"]
            elif i == 1:
                inc_date = PERSONA_2_KARIM["incorporated_on"]
            elif i == 2:
                inc_date = PERSONA_3_FASTFIX["incorporated_on"]

            businesses_rows.append(
                (
                    biz_id,
                    now_iso,
                    now_iso,
                    user_id,
                    pop["business_names"][i],
                    pop["sectors"][i],
                    pop["cities"][i],
                    pop["provinces"][i],
                    pop["legal_forms"][i],
                    inc_date.isoformat(),
                    pop["employees"][i],
                    pop["owner_gender"][i],
                    pop["age_bands"][i],
                    1 if pop["is_women_led"][i] else 0,
                    1 if pop["has_ntn"][i] else 0,
                    1,  # is_synthetic
                    int(pop["collateral_values"][i]),
                )
            )

            # Consent
            consents_rows.append(
                (
                    biz_id,
                    now_iso,
                    now_iso,
                    biz_id,
                    user_id,
                    "synthetic cashflow and bureau assessment",
                    "v1.0-synthetic",
                    now_iso,
                    None,
                )
            )

            # Application
            if i == 0:
                prod = PERSONA_1_AYESHA["product"]
                req_amt = PERSONA_1_AYESHA["amount_requested_pkr"]
                tenor = PERSONA_1_AYESHA["tenor_months"]
                purpose = PERSONA_1_AYESHA["purpose"]
            elif i == 1:
                prod = PERSONA_2_KARIM["product"]
                req_amt = PERSONA_2_KARIM["amount_requested_pkr"]
                tenor = PERSONA_2_KARIM["tenor_months"]
                purpose = PERSONA_2_KARIM["purpose"]
            elif i == 2:
                prod = PERSONA_3_FASTFIX["product"]
                req_amt = PERSONA_3_FASTFIX["amount_requested_pkr"]
                tenor = PERSONA_3_FASTFIX["tenor_months"]
                purpose = PERSONA_3_FASTFIX["purpose"]
            else:
                prod = products[i % len(products)]
                req_amt = int(
                    self.rng.choice(
                        [
                            300000,
                            500000,
                            800000,
                            1000000,
                            1500000,
                            2000000,
                            3000000,
                            5000000,
                        ]
                    )
                )
                tenor = int(self.rng.choice([6, 12, 18, 24]))
                purpose = f"Working capital expansion for {pop['sectors'][i]}"

            apps_rows.append(
                (
                    app_id,
                    now_iso,
                    now_iso,
                    biz_id,
                    prod,
                    req_amt,
                    tenor,
                    purpose,
                    enums.ApplicationStatus.SUBMITTED,
                    now_iso,
                    biz_id,
                )
            )

            # Primary Document (Bank Statement)
            statement_sha = hashlib.sha256(f"statement_biz_{biz_id}".encode()).hexdigest()
            doc_id = doc_id_counter
            doc_id_counter += 1

            parse_rep = '{"status": "parsed", "account_type": "current", "reconciled": true}'
            if pop["fraud_patterns"][i] == "tampered_balances":
                parse_rep = (
                    '{"status": "parsed", "account_type": "current", '
                    '"balance_discrepancy_pkr": 950000}'
                )

            docs_rows.append(
                (
                    doc_id,
                    now_iso,
                    now_iso,
                    biz_id,
                    enums.DocumentType.BANK_STATEMENT,
                    f"bank_statement_{biz_id}.csv",
                    statement_sha,
                    enums.ParseStatus.PARSED,
                    parse_rep,
                )
            )

            # Bureau Record (Messiness: missing for thin-file)
            if pop["is_thin_file"][i]:
                # 30% have a thin unscored record (< 12 months)
                if self.rng.uniform(0, 1) < 0.30 and i != 0:
                    bureau_rows.append(
                        (
                            bureau_id_counter,
                            now_iso,
                            now_iso,
                            biz_id,
                            doc_id,
                            None,  # score is None
                            0,  # active loans
                            0,  # delinquent
                            int(self.rng.integers(2, 10)),  # history < 12 mo
                        )
                    )
                    bureau_id_counter += 1
            else:
                # Thick file: score correlated with latent health
                base_score = int(620 + 50 * pop["latent_health"][i] + self.rng.normal(0, 25))
                score = int(np.clip(base_score, 450, 830))
                active_loans = int(self.rng.choice([0, 1, 2, 3], p=[0.4, 0.4, 0.15, 0.05]))
                delinquent = 1 if (score < 560 or self.rng.uniform(0, 1) < 0.05) else 0
                history_mo = int(self.rng.integers(14, 72))
                bureau_rows.append(
                    (
                        bureau_id_counter,
                        now_iso,
                        now_iso,
                        biz_id,
                        doc_id,
                        score,
                        active_loans,
                        delinquent,
                        history_mo,
                    )
                )
                bureau_id_counter += 1

            # 24 Monthly Utility Bill Payments (Messiness: late-arriving bills)
            util_h = pop["latent_health"][i]
            on_time_prob = float(expit(1.8 + 1.2 * util_h))
            cur_bill_date = START_DATE
            while cur_bill_date <= END_DATE:
                # Mid-month due date (15th)
                due_d = cur_bill_date.replace(day=15)
                # Punctuality
                if self.rng.uniform(0, 1) < on_time_prob:
                    paid_d: date | None = due_d - timedelta(days=int(self.rng.integers(0, 4)))
                else:
                    # Late payment or unpaid if very recent
                    if (END_DATE - due_d).days < 20 and self.rng.uniform(0, 1) < 0.4:
                        paid_d = None
                    else:
                        paid_d = due_d + timedelta(days=int(self.rng.integers(3, 28)))

                bill_amt = int(
                    np.clip(
                        15000 + 4000 * pop["employees"][i] + self.rng.normal(0, 3000),
                        5000,
                        500000,
                    )
                )
                bill_rows.append(
                    (
                        bill_id_counter,
                        now_iso,
                        now_iso,
                        biz_id,
                        doc_id,
                        enums.UtilityType.ELECTRICITY,
                        due_d.isoformat(),
                        paid_d.isoformat() if paid_d else None,
                        bill_amt,
                    )
                )
                bill_id_counter += 1

                # Advance month
                month = cur_bill_date.month + 1
                year = cur_bill_date.year
                if month > 12:
                    month = 1
                    year += 1
                cur_bill_date = date(year, month, 1)

            # Evaluation row (kept separate from model features!)
            eval_rows.append(
                (
                    biz_id,
                    biz_id,
                    app_id,
                    int(pop["default_12m"][i]),
                    float(pop["latent_health"][i]),
                    int(pop["is_synthetic_fraud"][i]),
                    pop["fraud_patterns"][i],
                    1 if pop["is_thin_file"][i] else 0,
                    now_iso,
                )
            )

        # Bulk insert initial relational records
        conn.executemany(
            "INSERT INTO users (id, created_at, updated_at, role) VALUES (?,?,?,?)",
            users_rows,
        )
        conn.executemany(
            """INSERT INTO businesses (
                id, created_at, updated_at, owner_id, name, sector, city, province,
                legal_form, incorporated_on, employees, owner_gender, owner_age_band,
                is_women_led, has_ntn, is_synthetic, collateral_value_pkr
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            businesses_rows,
        )
        conn.executemany(
            """INSERT INTO consent_records (
                id, created_at, updated_at, business_id, user_id, scope, policy_version,
                granted_at, revoked_at
            ) VALUES (?,?,?,?,?,?,?,?,?)""",
            consents_rows,
        )
        conn.executemany(
            """INSERT INTO applications (
                id, created_at, updated_at, business_id, product, amount_requested_pkr,
                tenor_months, purpose, status, submitted_at, consent_id
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            apps_rows,
        )
        conn.executemany(
            """INSERT INTO documents (
                id, created_at, updated_at, business_id, type, filename, sha256,
                parse_status, parse_report
            ) VALUES (?,?,?,?,?,?,?,?,?)""",
            docs_rows,
        )
        conn.executemany(
            """INSERT INTO bureau_records (
                id, created_at, updated_at, business_id, document_id, score,
                active_loans, delinquent_flag, history_months
            ) VALUES (?,?,?,?,?,?,?,?,?)""",
            bureau_rows,
        )
        conn.executemany(
            """INSERT INTO bill_payments (
                id, created_at, updated_at, business_id, document_id, utility_type,
                due_date, paid_date, amount_pkr
            ) VALUES (?,?,?,?,?,?,?,?,?)""",
            bill_rows,
        )
        conn.executemany(
            """INSERT INTO loan_evaluations (
                id, business_id, application_id, default_12m, latent_business_health,
                is_synthetic_fraud, fraud_pattern, is_thin_file, created_at
            ) VALUES (?,?,?,?,?,?,?,?,?)""",
            eval_rows,
        )
        conn.commit()

        # -------------------------------------------------------------------
        # 2. 24 Months of Daily Transactions with Balance Reconciliation
        # -------------------------------------------------------------------
        if not quiet:
            print("Generating 24 months of daily transactions with reconciled balances...")

        txn_csv_path = data_p / "transactions.csv"
        txn_csv_file = open(txn_csv_path, "w", newline="", encoding="utf-8")  # noqa: SIM115
        txn_writer = csv.writer(txn_csv_file)
        txn_writer.writerow(
            [
                "id",
                "business_id",
                "date",
                "amount_pkr",
                "direction",
                "channel",
                "category",
                "counterparty_hash",
                "balance_after",
            ]
        )

        # Inflow categories
        inflow_cats = [
            "customer_payment",
            "sales_deposit",
            "invoice_settlement",
            "pos_settlement",
            "digital_transfer_in",
        ]

        txn_id = 1
        reconciled_count = 0
        total_txns = 0

        # Persona sample transaction buffers
        persona_samples: dict[int, list[dict[str, Any]]] = {1: [], 2: [], 3: []}

        # Process businesses in batches for low memory footprint
        batch_size = 500
        for b_start in range(0, n, batch_size):
            b_end = min(b_start + batch_size, n)
            batch_txns = []

            for i in range(b_start, b_end):
                biz_id = i + 1
                h = pop["latent_health"][i]
                sector = pop["sectors"][i]
                emp = pop["employees"][i]
                fraud_pat = pop["fraud_patterns"][i]

                # Base daily revenue
                base_daily = max(
                    10000,
                    int((180000 + 40000 * emp) * np.exp(0.35 * h) / 30.0),
                )
                if biz_id == 1:
                    base_daily = 30000  # Ayesha: ~PKR 900k/mo
                elif biz_id == 2:
                    base_daily = 48000  # Karim: ~PKR 1.45M/mo
                elif biz_id == 3:
                    base_daily = 60000  # FastFix: ~PKR 1.8M/mo

                # Digital preference: shifted by health and sector
                p_digital = float(expit(0.3 + 0.6 * h))
                if sector == "IT and freelance services" or biz_id == 1:
                    p_digital = 0.85
                elif sector == "agri-processing":
                    p_digital = 0.25

                # Counterparties subset for this business
                biz_hash_subset = [
                    _HASH_POOL[(biz_id * 13 + k) % len(_HASH_POOL)] for k in range(15)
                ]

                # Simulation loop across 730 days
                cur_date = START_DATE
                cur_balance = int(base_daily * 15)  # Healthy starting buffer

                for day_idx in range(TOTAL_DAYS):
                    cal_month = cur_date.month
                    season_mult = get_sector_seasonality(sector, cal_month)
                    is_weekend = cur_date.weekday() >= 5
                    is_month_end = cur_date.day in (28, 29, 30, 31, 1)
                    is_mid_month = cur_date.day in (15, 16)

                    # Determine daily txns (1 primary txn per day + periodic bills/payroll)
                    # Outflow vs Inflow determination:
                    # Inflows happen on most sales days (~70% of days)
                    # Outflows happen on payroll, supplier replenishment days
                    is_outflow = False
                    category = inflow_cats[day_idx % len(inflow_cats)]
                    channel = (
                        self.rng.choice(
                            [
                                enums.Channel.BANK,
                                enums.Channel.WALLET,
                                enums.Channel.INSTANT_TRANSFER,
                            ]
                        )
                        if self.rng.uniform(0, 1) < p_digital
                        else enums.Channel.CASH_DEPOSIT
                    )

                    if is_month_end and day_idx % 2 == 0:
                        is_outflow = True
                        category = "payroll"
                        channel = enums.Channel.BANK
                        amount = int(np.clip((25000 + 5000 * (h > 0)) * emp / 4.0, 10000, 1000000))
                    elif is_mid_month and day_idx % 2 == 1:
                        is_outflow = True
                        category = "utility_bill"
                        channel = enums.Channel.INSTANT_TRANSFER
                        amount = int(np.clip(12000 + 3500 * emp, 5000, 200000))
                    elif day_idx % 4 == 0:
                        is_outflow = True
                        category = "supplier_payment"
                        channel = enums.Channel.BANK
                        amount = int(base_daily * season_mult * 1.8)
                    else:
                        is_outflow = False
                        category = inflow_cats[(day_idx + biz_id) % len(inflow_cats)]
                        is_retail_fb = sector in ("retail", "food and beverage")
                        weekend_boost = 1.3 if (is_weekend and is_retail_fb) else 1.0
                        amount = int(
                            max(
                                2000,
                                base_daily
                                * season_mult
                                * weekend_boost
                                * (0.8 + 0.4 * self.rng.uniform(0, 1)),
                            )
                        )

                    # Injected Fraud Patterns
                    if fraud_pat == "window_dressing":
                        # In the last 45 days, inject 3 massive round-trip deposits
                        if TOTAL_DAYS - day_idx in (40, 25, 10):
                            is_outflow = False
                            category = "digital_transfer_in"
                            channel = enums.Channel.INSTANT_TRANSFER
                            amount = int(base_daily * 25)  # Sudden huge inflow
                        elif TOTAL_DAYS - day_idx in (38, 23, 8):
                            is_outflow = True
                            category = "operational_expense"
                            channel = enums.Channel.BANK
                            amount = int(base_daily * 24)  # Immediate outflow
                    elif fraud_pat == "circular_flows":
                        if day_idx % 7 == 0:
                            # Rapid round-trip cycling between specific hashes
                            is_outflow = False
                            category = "invoice_settlement"
                            amount = int(base_daily * 8)
                        elif day_idx % 7 == 1:
                            is_outflow = True
                            category = "supplier_payment"
                            amount = int(base_daily * 7.9)

                    # Guard balance to ensure balance_after >= 0
                    if is_outflow:
                        if amount >= cur_balance:
                            # Constrain outflow to keep balance positive
                            amount = int(cur_balance * 0.7)
                        if amount == 0:
                            direction = enums.Direction.INFLOW
                            amount = int(base_daily * 0.5)
                            new_balance = cur_balance + amount
                        else:
                            direction = enums.Direction.OUTFLOW
                            new_balance = cur_balance - amount
                    else:
                        direction = enums.Direction.INFLOW
                        new_balance = cur_balance + amount

                    c_hash = biz_hash_subset[(day_idx + biz_id) % len(biz_hash_subset)]

                    # Mathematical balance reconciliation check
                    if direction == enums.Direction.INFLOW:
                        assert new_balance == cur_balance + amount
                    else:
                        assert new_balance == cur_balance - amount
                    assert new_balance >= 0
                    reconciled_count += 1
                    total_txns += 1

                    row_tuple = (
                        txn_id,
                        now_iso,
                        now_iso,
                        biz_id,
                        cur_date.isoformat(),
                        amount,
                        direction,
                        channel,
                        category,
                        c_hash,
                        new_balance,
                        biz_id,  # doc_id matches primary doc
                    )
                    batch_txns.append(row_tuple)

                    # Write to CSV
                    txn_writer.writerow(
                        [
                            txn_id,
                            biz_id,
                            cur_date.isoformat(),
                            amount,
                            direction,
                            channel,
                            category,
                            c_hash,
                            new_balance,
                        ]
                    )

                    # Save persona samples (first 3 businesses)
                    if biz_id in (1, 2, 3):
                        persona_samples[biz_id].append(
                            {
                                "date": cur_date.isoformat(),
                                "amount_pkr": amount,
                                "direction": direction,
                                "channel": channel,
                                "category": category,
                                "counterparty_hash": c_hash,
                                "balance_pkr": new_balance,
                            }
                        )

                    cur_balance = new_balance
                    txn_id += 1
                    cur_date += timedelta(days=1)

            # Insert batch into SQLite
            conn.executemany(
                """INSERT INTO transactions (
                    id, created_at, updated_at, business_id, date, amount_pkr,
                    direction, channel, category, counterparty_hash, balance_after,
                    document_id
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                batch_txns,
            )
            conn.commit()

        txn_csv_file.close()

        # -------------------------------------------------------------------
        # 3. Export CSV copies to /data
        # -------------------------------------------------------------------
        if not quiet:
            print("Exporting relational CSV copies to data/ ...")

        # businesses.csv
        with open(data_p / "businesses.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "id",
                    "name",
                    "sector",
                    "city",
                    "province",
                    "legal_form",
                    "employees",
                    "owner_gender",
                    "owner_age_band",
                    "is_women_led",
                    "has_ntn",
                    "is_synthetic",
                    "collateral_value_pkr",
                ]
            )
            for r in businesses_rows:
                w.writerow(
                    [
                        r[0],
                        r[4],
                        r[5],
                        r[6],
                        r[7],
                        r[8],
                        r[10],
                        r[11],
                        r[12],
                        bool(r[13]),
                        bool(r[14]),
                        bool(r[15]),
                        r[16],
                    ]
                )

        # applications.csv
        with open(data_p / "applications.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "id",
                    "business_id",
                    "product",
                    "amount_requested_pkr",
                    "tenor_months",
                    "purpose",
                    "status",
                ]
            )
            for r in apps_rows:
                w.writerow([r[0], r[3], r[4], r[5], r[6], r[7], r[8]])

        # bureau_records.csv
        with open(data_p / "bureau_records.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "id",
                    "business_id",
                    "score",
                    "active_loans",
                    "delinquent_flag",
                    "history_months",
                ]
            )
            for r in bureau_rows:
                w.writerow([r[0], r[3], r[5], r[6], bool(r[7]), r[8]])

        # bill_payments.csv
        with open(data_p / "bill_payments.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "id",
                    "business_id",
                    "utility_type",
                    "due_date",
                    "paid_date",
                    "amount_pkr",
                ]
            )
            for r in bill_rows:
                w.writerow([r[0], r[3], r[5], r[6], r[7], r[8]])

        # loan_evaluations.csv
        with open(data_p / "loan_evaluations.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "id",
                    "business_id",
                    "application_id",
                    "default_12m",
                    "latent_business_health",
                    "is_synthetic_fraud",
                    "fraud_pattern",
                    "is_thin_file",
                ]
            )
            for r in eval_rows:
                w.writerow([r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]])

        conn.close()

        # -------------------------------------------------------------------
        # 4. Generate Demo Sample Statements (CSV & text-based PDF in /samples)
        # -------------------------------------------------------------------
        if not quiet:
            print("Generating sample bank statements (CSV + PDF) in samples/ ...")

        self._generate_sample_statements(samples_p, persona_samples)

        # -------------------------------------------------------------------
        # 5. Calculate & Print Acceptance Summary Statistics
        # -------------------------------------------------------------------
        default_rate = float(np.mean(pop["default_12m"]))
        thin_file_share = float(np.mean(pop["is_thin_file"]))
        women_led_share = float(np.mean(pop["is_women_led"]))
        ntn_share = float(np.mean(pop["has_ntn"]))
        fraud_share = float(np.mean(pop["is_synthetic_fraud"]))
        recon_pct = (reconciled_count / total_txns) * 100.0 if total_txns > 0 else 0.0

        summary = {
            "n": n,
            "seed": self.seed,
            "default_rate": default_rate,
            "thin_file_share": thin_file_share,
            "women_led_share": women_led_share,
            "ntn_share": ntn_share,
            "fraud_share": fraud_share,
            "total_transactions": total_txns,
            "balance_reconciliation_pct": recon_pct,
            "personas": {
                "persona_1": PERSONA_1_AYESHA["name"],
                "persona_2": PERSONA_2_KARIM["name"],
                "persona_3": PERSONA_3_FASTFIX["name"],
            },
            "db_path": str(db_p),
            "data_dir": str(data_p),
            "samples_dir": str(samples_p),
        }

        print("\n=======================================================")
        print("          SME CREDIT SYNTHESIS SUMMARY")
        print("=======================================================")
        print(f"Population Size (N):           {n}")
        print(f"Random Seed:                   {self.seed}")
        print(f"Women-Led Share:               {women_led_share * 100:.1f}%  (target: ~20%)")
        print(f"Thin-File Share:               {thin_file_share * 100:.1f}%  (target: ~35%)")
        print(f"NTN Registered Share:          {ntn_share * 100:.1f}%  (target: ~60%)")
        print(f"12-Month Default Rate:         {default_rate * 100:.1f}%  (target: 8-12%)")
        print(f"Synthetic Fraud Share:         {fraud_share * 100:.1f}%  (target: 3-5%)")
        print(f"Total Daily Transactions:      {total_txns:,}")
        print(f"Balance Reconciliation Rate:   {recon_pct:.1f}%")
        print("-------------------------------------------------------")
        n1 = PERSONA_1_AYESHA["name"]
        n2 = PERSONA_2_KARIM["name"]
        n3 = PERSONA_3_FASTFIX["name"]
        print(f"  [ID 1] {n1:<20} Thin-file: YES | Women-led: YES | Default: NO")
        print(f"  [ID 2] {n2:<20} Thin-file: NO  | Borderline: YES | Default: NO")
        print(f"  [ID 3] {n3:<20} Window Dressing Fraud: YES   | Default: YES")
        print("-------------------------------------------------------")
        print(f"Database Output:  {db_p}")
        print(f"CSV Output:       {data_p}")
        print(f"Samples Output:   {samples_p}")
        print("=======================================================\n")

        return summary

    def _generate_sample_statements(
        self, samples_dir: Path, persona_samples: dict[int, list[dict[str, Any]]]
    ) -> None:
        """Create sample statement CSV and text-based PDF for each persona."""
        persona_meta = {
            1: {
                "name": PERSONA_1_AYESHA["name"],
                "slug": "ayeshas_boutique",
                "bank": "Meezan Bank Limited — Islamic Banking",
                "iban": "PK36MEZN0001928374650101",
            },
            2: {
                "name": PERSONA_2_KARIM["name"],
                "slug": "karim_traders",
                "bank": "Habib Bank Limited (HBL)",
                "iban": "PK45HABB0002837465910202",
            },
            3: {
                "name": PERSONA_3_FASTFIX["name"],
                "slug": "fastfix_logistics",
                "bank": "Bank Alfalah Limited",
                "iban": "PK12ALFH0003948572610303",
            },
        }

        for pid, pdata in persona_meta.items():
            txns = persona_samples[pid]
            slug = pdata["slug"]

            # 1. Statement CSV
            csv_path = samples_dir / f"{slug}_statement.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(
                    [
                        "date",
                        "description",
                        "category",
                        "channel",
                        "counterparty_hash",
                        "inflow_pkr",
                        "outflow_pkr",
                        "balance_pkr",
                    ]
                )
                for t in txns:
                    inflow = t["amount_pkr"] if t["direction"] == enums.Direction.INFLOW else 0
                    outflow = t["amount_pkr"] if t["direction"] == enums.Direction.OUTFLOW else 0
                    w.writerow(
                        [
                            t["date"],
                            f"{t['category'].replace('_', ' ').title()} via {t['channel']}",
                            t["category"],
                            t["channel"],
                            t["counterparty_hash"],
                            inflow,
                            outflow,
                            t["balance_pkr"],
                        ]
                    )

            # 2. Text-based PDF statement
            pdf_path = samples_dir / f"{slug}_statement.pdf"
            self._write_text_pdf(
                pdf_path=pdf_path,
                bank_name=pdata["bank"],
                account_title=pdata["name"],
                iban=pdata["iban"],
                txns=txns[-45:],  # Last 45 days for crisp statement view
            )

    def _write_text_pdf(
        self,
        pdf_path: Path,
        bank_name: str,
        account_title: str,
        iban: str,
        txns: list[dict[str, Any]],
    ) -> None:
        """Construct valid text-based PDF 1.4 statement with no external dependencies."""
        lines = [
            "================================================================================",
            f"  {bank_name.upper()}",
            "  OFFICIAL STATEMENT OF ACCOUNT — SYNTHETIC CASH FLOW AUDIT",
            "================================================================================",
            f"Account Title:   {account_title}",
            f"IBAN:            {iban}",
            "Currency:        PKR (Pakistani Rupee)",
            "Statement Span:  2024-10-01 to 2026-09-30 (Recent transactions shown below)",
            "Notice:          100% Synthetic Data generated for AI credit scrutiny demo.",
            "--------------------------------------------------------------------------------",
            "DATE        | CHANNEL       | CATEGORY          | INFLOW     | OUTFLOW    | BALANCE",
            "--------------------------------------------------------------------------------",
        ]

        for t in txns:
            d_str = t["date"]
            ch_str = f"{t['channel'][:13]:<13}"
            cat_str = f"{t['category'][:17]:<17}"
            inflow = (
                f"{t['amount_pkr']:>10,}"
                if t["direction"] == enums.Direction.INFLOW
                else "         0"
            )
            outflow = (
                f"{t['amount_pkr']:>10,}"
                if t["direction"] == enums.Direction.OUTFLOW
                else "         0"
            )
            bal = f"{t['balance_pkr']:>10,}"
            lines.append(f"{d_str}  | {ch_str} | {cat_str} | {inflow} | {outflow} | {bal}")

        lines.extend(
            [
                "--------------------------------------------------------------------------------",
                f"Closing Balance PKR: {txns[-1]['balance_pkr']:,}",
                "End of Statement — All running balances mathematically reconciled.",
                "================================================================================",
            ]
        )

        # Assemble PDF stream operators
        stream_cmds = [
            "BT",
            "/F1 9 Tf",
            "40 760 Td",
            "12 TL",
        ]
        for line_item in lines:
            escaped = line_item.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            stream_cmds.append(f"({escaped}) Tj")
            stream_cmds.append("T*")
        stream_cmds.append("ET")

        stream_bytes = "\n".join(stream_cmds).encode("latin-1", "replace")

        # PDF 1.4 objects
        objects = []
        objects.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
        objects.append(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n")
        objects.append(
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> "
            b"/MediaBox [0 0 612 792] /Contents 5 0 R >>\nendobj\n"
        )
        objects.append(b"4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>\nendobj\n")
        objects.append(
            f"5 0 obj\n<< /Length {len(stream_bytes)} >>\nstream\n".encode("latin-1")
            + stream_bytes
            + b"\nendstream\nendobj\n"
        )

        body = b"".join(objects)
        header = b"%PDF-1.4\n"
        xref_pos = len(header) + len(body)

        offsets = [0]
        cur = len(header)
        for obj in objects:
            offsets.append(cur)
            cur += len(obj)

        xref = [f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("latin-1")]
        for off in offsets[1:]:
            xref.append(f"{off:010d} 00000 n \n".encode("latin-1"))

        xref_bytes = b"".join(xref)
        trailer = (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_pos}\n%%EOF\n".encode("latin-1")
        )

        with open(pdf_path, "wb") as f:
            f.write(header + body + xref_bytes + trailer)


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic Pakistani SME credit dataset with latent health, "
            "transactions, and personas."
        )
    )
    parser.add_argument(
        "--n",
        type=int,
        default=3000,
        help="Number of SMEs to generate (default: 3000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for exact reproducibility (default: 42)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Target SQLite database file path (default: backend/sme_credit.db or ./sme_credit.db)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory to save CSV exports (default: data)",
    )
    parser.add_argument(
        "--samples-dir",
        type=str,
        default=None,
        help="Directory to save statement samples (default: samples)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress intermediate logging",
    )

    args = parser.parse_args()

    # Resolve paths relative to repo root
    current = Path.cwd()
    if (current / "backend").is_dir():
        repo_root = current
    elif current.name == "backend":
        repo_root = current.parent
    else:
        repo_root = current

    db_path = args.db_path or str(repo_root / "backend" / "sme_credit.db")
    data_dir = args.data_dir or str(repo_root / "data")
    samples_dir = args.samples_dir or str(repo_root / "samples")

    generator = SyntheticDataGenerator(n=args.n, seed=args.seed)
    generator.generate_all(
        db_path=db_path,
        data_dir=data_dir,
        samples_dir=samples_dir,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
