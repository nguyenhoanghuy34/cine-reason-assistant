from typing import Literal

from pydantic import BaseModel


class IntentClassification(BaseModel):
    intent: Literal[
        "GENERAL",
        "PERSONAL",
        "RELATED_USERS",
        "OTHER",
    ]

    reason: str