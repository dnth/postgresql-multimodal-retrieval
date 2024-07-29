from decimal import Decimal

from dataclasses import dataclass


@dataclass
class Result:
    id: int
    image_filename: str
    rrf_score: Decimal
