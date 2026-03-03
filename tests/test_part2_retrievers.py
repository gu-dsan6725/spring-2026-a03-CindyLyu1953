"""Test cases for Part 2: Retrievers (CSV and text)."""

import pytest
from pathlib import Path

from src.part2_retrievers import (
    load_csv,
    retrieve_csv_revenue_by_category_month,
    retrieve_csv_region_highest_volume,
    retrieve_csv_sales_summary,
    retrieve_csv_region_product_sales,
    load_product_pages,
    build_bm25_index,
    retrieve_text_bm25,
    get_product_rating,
    retrieve_text_best_reviews,
    PRODUCT_IDS,
)


@pytest.fixture
def csv_path():
    return Path(__file__).resolve().parents[1] / "data" / "structured" / "daily_sales.csv"


@pytest.fixture
def text_dir():
    return Path(__file__).resolve().parents[1] / "data" / "unstructured"


class TestLoadCsv:
    """Test CSV loading."""

    def test_load_csv_exists(self, csv_path):
        if not csv_path.exists():
            pytest.skip("daily_sales.csv not found")
        df = load_csv(csv_path)
        assert len(df) > 0
        assert "date" in df.columns
        assert "product_id" in df.columns
        assert "total_revenue" in df.columns
        assert "region" in df.columns

    def test_load_csv_date_parsed(self, csv_path):
        if not csv_path.exists():
            pytest.skip("daily_sales.csv not found")
        df = load_csv(csv_path)
        assert "datetime" in str(df["date"].dtype)


class TestCsvRetrieval:
    """Test CSV retrieval functions."""

    def test_revenue_by_category_month(self, csv_path):
        if not csv_path.exists():
            pytest.skip("daily_sales.csv not found")
        df = load_csv(csv_path)
        rev = retrieve_csv_revenue_by_category_month(df, "Electronics", 2024, 12)
        assert isinstance(rev, (int, float))
        assert rev >= 0

    def test_region_highest_volume(self, csv_path):
        if not csv_path.exists():
            pytest.skip("daily_sales.csv not found")
        df = load_csv(csv_path)
        region = retrieve_csv_region_highest_volume(df)
        assert region in ["North", "South", "East", "West", "Central"]

    def test_sales_summary_non_empty(self, csv_path):
        if not csv_path.exists():
            pytest.skip("daily_sales.csv not found")
        df = load_csv(csv_path)
        summary = retrieve_csv_sales_summary(df)
        assert "Sales by product" in summary
        assert "Sales by region" in summary

    def test_region_product_sales(self, csv_path):
        if not csv_path.exists():
            pytest.skip("daily_sales.csv not found")
        df = load_csv(csv_path)
        result = retrieve_csv_region_product_sales(df, "West")
        assert "product_id" in result.columns
        assert "units_sold" in result.columns


class TestLoadProductPages:
    """Test product page loading."""

    def test_load_pages_exists(self, text_dir):
        if not text_dir.exists():
            pytest.skip("data/unstructured not found")
        pages = load_product_pages(text_dir)
        assert len(pages) > 0
        for pid, content in pages:
            assert isinstance(pid, str)
            assert isinstance(content, str)
            assert len(content) > 0


class TestBM25Retrieval:
    """Test BM25 text retrieval."""

    def test_build_index_and_retrieve(self, text_dir):
        if not text_dir.exists():
            pytest.skip("data/unstructured not found")
        pages = load_product_pages(text_dir)
        bm25, _ = build_bm25_index(pages)
        result = retrieve_text_bm25(pages, bm25, "Wireless Bluetooth Headphones", top_k=2)
        assert "Wireless" in result or "Bluetooth" in result or "ELEC001" in result

    def test_get_product_rating(self):
        content = "Average Rating: 4.4/5 (2,847 reviews)"
        assert get_product_rating(content) == 4.4

    def test_get_product_rating_missing(self):
        assert get_product_rating("no rating here") == 0.0

    def test_retrieve_best_reviews(self, text_dir):
        if not text_dir.exists():
            pytest.skip("data/unstructured not found")
        pages = load_product_pages(text_dir)
        result = retrieve_text_best_reviews(pages, top_k=2)
        assert len(result) > 0


class TestProductIds:
    """Test product ID mapping."""

    def test_product_ids_has_expected_entries(self):
        assert "ELEC001" in PRODUCT_IDS
        assert PRODUCT_IDS["ELEC001"] == "Wireless Bluetooth Headphones"
        assert "HOME003" in PRODUCT_IDS
        assert PRODUCT_IDS["HOME003"] == "Air Fryer 5.5L"
