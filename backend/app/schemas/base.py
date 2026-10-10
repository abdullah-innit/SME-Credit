"""Contract configuration and structural privacy validation."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

PROHIBITED_KEYS = {
    "account_number",
    "raw_account_number",
    "iban",
    "account_no",
    "counterparty_name",
    "counterparty",
    "raw_counterparty_name",
}


def validate_private_keys(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_").replace(" ", "_")
            if normalized in PROHIBITED_KEYS:
                raise ValueError("Raw account/counterparty identifiers must not be persisted")
            validate_private_keys(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            validate_private_keys(item)


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", from_attributes=True)

    @model_validator(mode="before")
    @classmethod
    def reject_private_keys(cls, value: Any) -> Any:
        validate_private_keys(value)
        return value


class ReadContract(Contract):
    id: int
    created_at: datetime
    updated_at: datetime
