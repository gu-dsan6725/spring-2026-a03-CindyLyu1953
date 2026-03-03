"""Part 2: Query router for multi-source RAG (CSV vs text vs both)."""

from enum import Enum


class SourceRoute(str, Enum):
    """Route to data source(s)."""

    CSV_ONLY = "csv_only"
    TEXT_ONLY = "text_only"
    BOTH = "both"


def route_question(question: str) -> SourceRoute:
    """
    Classify question to determine which data source(s) to query.
    Uses keyword heuristics.
    """
    q = question.lower()

    # CSV indicators: revenue, sales, region, December, category, volume
    csv_keywords = [
        "revenue",
        "sales",
        "region",
        "december",
        "category",
        "volume",
        "total",
        "highest",
        "lowest",
        "how many",
        "how much",
        "units sold",
    ]

    # Text indicators: features, reviews, customers say, product details
    text_keywords = [
        "features",
        "key features",
        "reviews",
        "customers say",
        "customer",
        "product details",
        "product page",
        "ease of",
        "cleaning",
        "description",
    ]

    # Both: recommend + rated/selling, best reviews + selling
    both_keywords = [
        "best",
        "recommend",
        "highly rated",
        "sells well",
        "selling well",
        "rated and",
        "reviews and",
        "fitness",
        "product for",
    ]

    csv_match = sum(1 for k in csv_keywords if k in q)
    text_match = sum(1 for k in text_keywords if k in q)
    both_match = sum(1 for k in both_keywords if k in q)

    # Both takes precedence if multiple signals
    if both_match >= 1 or (csv_match >= 1 and text_match >= 1):
        return SourceRoute.BOTH
    if text_match > csv_match:
        return SourceRoute.TEXT_ONLY
    if csv_match >= 1:
        return SourceRoute.CSV_ONLY

    # Default: if mentions product name or reviews, try text; else both
    if "product" in q or "review" in q:
        return SourceRoute.TEXT_ONLY
    return SourceRoute.BOTH
