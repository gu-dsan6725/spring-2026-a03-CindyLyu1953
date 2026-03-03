"""Part 2: Retrievers for CSV and unstructured product text."""

import re
from pathlib import Path

import pandas as pd
from rank_bm25 import BM25Okapi

# Product ID to name mapping (from generate_data.py)
PRODUCT_IDS = {
    "ELEC001": "Wireless Bluetooth Headphones",
    "HOME003": "Air Fryer 5.5L",
    "SPRT001": "Yoga Mat Premium",
    "BEAU001": "Vitamin C Serum",
    "CLTH001": "Running Shoes Men",
    "BOOK001": "Python Programming Guide",
    "TOYS001": "Building Blocks Set 500pc",
    "OFFC001": "Ergonomic Office Chair",
    "PETS001": "Dog Food Premium 10kg",
    "FOOD001": "Organic Coffee Beans 1kg",
}
PRODUCT_NAMES = {v: k for k, v in PRODUCT_IDS.items()}


def load_csv(csv_path: Path) -> pd.DataFrame:
    """Load daily_sales.csv with proper types."""
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def retrieve_csv_revenue_by_category_month(
    df: pd.DataFrame, category: str, year: int, month: int
) -> float:
    """Sum total_revenue for a category in a given month."""
    mask = (
        (df["category"] == category)
        & (df["date"].dt.year == year)
        & (df["date"].dt.month == month)
    )
    return df.loc[mask, "total_revenue"].sum()


def retrieve_csv_region_highest_volume(df: pd.DataFrame) -> str:
    """Return region with highest total units_sold."""
    by_region = df.groupby("region")["units_sold"].sum()
    return by_region.idxmax()


def retrieve_csv_sales_summary(df: pd.DataFrame) -> str:
    """Return a summary of sales by product and region for multi-source queries."""
    by_product = df.groupby("product_id").agg(
        units_sold=("units_sold", "sum"),
        revenue=("total_revenue", "sum"),
    ).reset_index()
    by_region = df.groupby("region")["units_sold"].sum().reset_index()
    return f"Sales by product:\n{by_product.to_string()}\n\nSales by region:\n{by_region.to_string()}"


def retrieve_csv_region_product_sales(
    df: pd.DataFrame, region: str
) -> pd.DataFrame:
    """Get sales by product for a specific region."""
    mask = df["region"] == region
    return df.loc[mask].groupby("product_id").agg(
        units_sold=("units_sold", "sum"),
        revenue=("total_revenue", "sum"),
    ).reset_index()


def load_product_pages(text_dir: Path) -> list[tuple[str, str]]:
    """Load all product page files, return list of (product_id, content)."""
    pages = []
    for f in text_dir.glob("*_product_page.txt"):
        pid = f.stem.replace("_product_page", "")
        content = f.read_text(encoding="utf-8")
        pages.append((pid, content))
    return pages


def _tokenize(text: str) -> list[str]:
    """Simple tokenizer for BM25."""
    return re.findall(r"\w+", text.lower())


def build_bm25_index(pages: list[tuple[str, str]]) -> BM25Okapi:
    """Build BM25 index over product pages."""
    corpus = [content for _, content in pages]
    tokenized = [_tokenize(c) for c in corpus]
    return BM25Okapi(tokenized), pages


def retrieve_text_bm25(
    pages: list[tuple[str, str]],
    bm25: BM25Okapi,
    query: str,
    top_k: int = 2,
) -> str:
    """Retrieve top-k product pages by BM25 score."""
    tokenized = [_tokenize(c) for _, c in pages]
    scores = bm25.get_scores(_tokenize(query))
    top_indices = scores.argsort()[-top_k:][::-1]
    parts = []
    for i in top_indices:
        pid, content = pages[i]
        name = PRODUCT_IDS.get(pid, pid)
        parts.append(f"--- {name} ({pid}) ---\n{content[:3000]}")
    return "\n\n".join(parts)


def retrieve_text_simple_search(
    pages: list[tuple[str, str]], query: str, top_k: int = 2
) -> str:
    """Fallback: retrieve by keyword match (no BM25)."""
    q_lower = query.lower()
    scored = []
    for pid, content in pages:
        content_lower = content.lower()
        score = sum(1 for w in q_lower.split() if w in content_lower)
        scored.append((score, pid, content))
    scored.sort(key=lambda x: -x[0])
    parts = []
    for _, pid, content in scored[:top_k]:
        name = PRODUCT_IDS.get(pid, pid)
        parts.append(f"--- {name} ({pid}) ---\n{content[:3000]}")
    return "\n\n".join(parts)


def get_product_rating(content: str) -> float:
    """Extract average rating from product page (e.g. 'Average Rating: 4.4/5')."""
    m = re.search(r"Average Rating:\s*([\d.]+)", content, re.I)
    return float(m.group(1)) if m else 0.0


def retrieve_text_best_reviews(
    pages: list[tuple[str, str]], top_k: int = 3
) -> str:
    """Retrieve product pages with highest average ratings (for 'best reviews' queries)."""
    rated = [(get_product_rating(c), pid, c) for pid, c in pages]
    rated.sort(key=lambda x: -x[0])
    parts = []
    for _, pid, content in rated[:top_k]:
        name = PRODUCT_IDS.get(pid, pid)
        parts.append(f"--- {name} ({pid}) ---\n{content[:2500]}")
    return "\n\n".join(parts)
