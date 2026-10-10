"""Canonical string enums shared by persistence and API contracts."""

from enum import StrEnum


class UserRole(StrEnum):
    APPLICANT = "applicant"
    OFFICER = "officer"
    ADMIN = "admin"


class Product(StrEnum):
    WORKING_CAPITAL = "working_capital"
    TERM_LOAN = "term_loan"
    INVOICE_FINANCING = "invoice_financing"


class ApplicationStatus(StrEnum):
    DRAFT = "draft"
    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"


class Direction(StrEnum):
    INFLOW = "inflow"
    OUTFLOW = "outflow"


class Channel(StrEnum):
    BANK = "bank"
    WALLET = "wallet"
    CASH_DEPOSIT = "cash_deposit"
    INSTANT_TRANSFER = "instant_transfer"


class DocumentType(StrEnum):
    BANK_STATEMENT = "bank_statement"
    WALLET_STATEMENT = "wallet_statement"
    UTILITY_BILL = "utility_bill"
    BUREAU_REPORT = "bureau_report"
    BUSINESS_REGISTRATION = "business_registration"
    OTHER = "other"


class ParseStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    PARSED = "parsed"
    FAILED = "failed"


class UtilityType(StrEnum):
    ELECTRICITY = "electricity"
    GAS = "gas"
    WATER = "water"
    INTERNET = "internet"
    OTHER = "other"


class Severity(StrEnum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Outcome(StrEnum):
    APPROVE = "approve"
    REJECT = "reject"
    REFER = "refer"


class RiskTier(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Gender(StrEnum):
    FEMALE = "female"
    MALE = "male"
    OTHER = "other"
    UNDISCLOSED = "undisclosed"


class AgeBand(StrEnum):
    UNDER_25 = "under_25"
    AGE_25_34 = "25_34"
    AGE_35_44 = "35_44"
    AGE_45_54 = "45_54"
    AGE_55_PLUS = "55_plus"
    UNDISCLOSED = "undisclosed"
    # Compatibility member names share the canonical values.
    BAND_25_34 = "25_34"
    BAND_35_44 = "35_44"
    BAND_45_54 = "45_54"
    BAND_55_PLUS = "55_plus"

    @classmethod
    def _missing_(cls, value: object) -> "AgeBand | None":
        # Accept legacy input such as 35-44 but persist only canonical 35_44.
        if isinstance(value, str):
            return cls._value2member_map_.get(value.replace("-", "_"))
        return None


class LegalForm(StrEnum):
    SOLE_PROPRIETORSHIP = "sole_proprietorship"
    PARTNERSHIP = "partnership"
    PRIVATE_LIMITED = "private_limited"
    PUBLIC_LIMITED = "public_limited"
    OTHER = "other"


class AiCallStatus(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"
    OFFLINE = "offline"


class AppealStatus(StrEnum):
    PENDING = "pending"
    IN_REVIEW = "in_review"
    RESOLVED = "resolved"


# Older callers may use this name; this is the same enum, not a second definition.
OwnerAgeBand = AgeBand
