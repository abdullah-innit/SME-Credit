# ruff: noqa: E501
"""Feature registry used by the model and counterfactual planner."""

from dataclasses import dataclass
from typing import Literal

Direction = Literal["better", "worse", "none"]
FEATURE_VERSION = "1.0.0"


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    description: str
    unit: str
    group: str
    monotonic: Direction
    actionable: bool
    bounds: tuple[float, float]
    effort_weight: float
    time_to_effect_months: int
    display_en: str
    display_ur: str


# name | description | unit | group | direction | actionable | low | high | effort | months | English | Urdu
_ROWS = """
avg_monthly_inflow|Mean monthly inflow over six months|PKR/month|cash_flow|better|1|0|100000000|5|6|Monthly inflow|ماہانہ آمدنی
inflow_cv|Monthly inflow standard deviation divided by mean|ratio|cash_flow|worse|1|0|5|4|6|Inflow volatility|آمدنی میں اتار چڑھاؤ
revenue_trend_slope|OLS monthly inflow slope over six months|PKR/month|cash_flow|better|1|-10000000|10000000|5|6|Revenue trend|آمدنی کا رجحان
negative_cash_flow_months|Months with outflow above inflow|months|cash_flow|worse|1|0|6|4|3|Negative cash flow months|منفی نقد بہاؤ والے ماہ
avg_balance_ratio|Mean observed closing balance over monthly inflow|ratio|liquidity|better|1|0|100|4|3|Average balance ratio|اوسط بقایا تناسب
min_balance_ratio|Minimum observed closing balance over monthly inflow|ratio|liquidity|better|1|0|100|5|3|Minimum balance ratio|کم ترین بقایا تناسب
dscr_proxy|Monthly net cash flow over known debt service plus proposed principal instalment|ratio|affordability|better|1|-100|100|5|1|Debt coverage proxy|قرض ادائیگی کی گنجائش
instalment_to_inflow|Proposed principal instalment over monthly inflow|ratio|affordability|worse|1|0|100|2|0|Instalment to inflow|قسط بمقابلہ آمدنی
amount_to_annual_revenue|Requested principal over annualised inflow|ratio|affordability|worse|1|0|100|2|0|Loan to annual revenue|قرض بمقابلہ سالانہ آمدنی
top3_customer_concentration|Top three inflow counterparties share|share|concentration|worse|1|0|1|5|6|Top three customer share|تین بڑے گاہکوں کا حصہ
digital_payment_share|Noncash inflow share by amount|share|channels|better|1|0|1|3|2|Digital payment share|ڈیجیٹل ادائیگیوں کا حصہ
cash_deposit_share|Cash-deposit inflow share by amount|share|channels|worse|1|0|1|3|2|Cash deposit share|نقد جمع کا حصہ
active_day_ratio|Transaction days over observed calendar days|share|activity|better|1|0|1|4|3|Active days|فعال دن
seasonality_index|Peak monthly inflow over average inflow|ratio|cash_flow|worse|1|0|6|5|6|Seasonality index|موسمی تبدیلی کا اشاریہ
on_time_bill_ratio|Bills paid by due date over bills due|share|bills|better|1|0|1|2|1|On-time bill share|بروقت بلوں کا حصہ
avg_days_late|Mean days late for due bills|days|bills|worse|1|0|365|2|1|Average days late|اوسط تاخیر کے دن
vintage_months|Completed months since incorporation|months|profile|better|0|0|600|0|0|Business age|کاروبار کی عمر
has_ntn|Tax registration indicator|flag|profile|better|1|0|1|4|2|NTN registered|این ٹی این رجسٹرڈ
bureau_score|Latest bureau score or neutral 600|score|bureau|better|0|0|1000|0|0|Bureau score|کریڈٹ بیورو اسکور
delinquency_flag|Latest bureau delinquency indicator|flag|bureau|worse|0|0|1|0|0|Delinquency|بقایا ادائیگی
active_loan_count|Latest reported active loans|count|bureau|worse|0|0|100|0|0|Active loans|فعال قرضے
employees|Reported employees|count|profile|none|0|0|10000|0|0|Employees|ملازمین
collateral_coverage|Collateral value over requested principal|ratio|security|better|1|0|100|5|3|Collateral coverage|ضمانت کا تناسب
has_bureau_record|Bureau record available by application date|flag|missingness|none|0|0|1|0|0|Bureau record available|بیورو ریکارڈ دستیاب
has_transaction_history|Pre-application transaction exists|flag|missingness|none|0|0|1|0|0|Transaction history available|لین دین کی تاریخ دستیاب
has_bill_history|Pre-application bill due|flag|missingness|none|0|0|1|0|0|Bill history available|بلوں کی تاریخ دستیاب
has_balance_data|Pre-application closing balance exists|flag|missingness|none|0|0|1|0|0|Balance data available|بقایا معلومات دستیاب
has_debt_service_data|Existing monthly debt service explicitly supplied|flag|missingness|none|0|0|1|0|0|Debt service known|قرض کی ادائیگی معلوم
transaction_count|Trailing six-month transaction count|count|activity|none|0|0|100000|0|0|Transactions|لین دین
avg_monthly_net_cash_flow|Mean monthly inflow less outflow|PKR/month|cash_flow|better|1|-100000000|100000000|5|3|Monthly net cash flow|ماہانہ خالص نقد بہاؤ
"""


def _parse(line: str) -> FeatureSpec:
    n, d, u, g, m, a, lo, hi, e, t, en, ur = line.split("|")
    return FeatureSpec(
        n, d, u, g, m, bool(int(a)), (float(lo), float(hi)), float(e), int(t), en, ur
    )


FEATURES = {spec.name: spec for spec in map(_parse, _ROWS.strip().splitlines())}
MODEL_FEATURES = tuple(FEATURES)


def validate_registry(model_features: tuple[str, ...] = MODEL_FEATURES) -> None:
    missing = set(model_features) - FEATURES.keys()
    if missing:
        raise ValueError(f"Model features missing metadata: {sorted(missing)}")
    for name in model_features:
        spec = FEATURES[name]
        if not all((spec.description, spec.unit, spec.group, spec.display_en, spec.display_ur)):
            raise ValueError(f"Incomplete metadata: {name}")
        if (
            spec.bounds[0] > spec.bounds[1]
            or spec.effort_weight < 0
            or spec.time_to_effect_months < 0
        ):
            raise ValueError(f"Invalid metadata: {name}")
    if any("gender" in n or "age_band" in n for n in model_features):
        raise ValueError("Protected attributes cannot be model features")


validate_registry()
