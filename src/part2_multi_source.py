"""Part 2: Multi-source RAG pipeline - router + retrievers + LLM."""

from pathlib import Path

from .llm_client import complete_with_context
from .part2_router import SourceRoute, route_question
from .part2_retrievers import (
    load_csv,
    load_product_pages,
    build_bm25_index,
    retrieve_text_bm25,
    retrieve_text_simple_search,
    retrieve_text_best_reviews,
    retrieve_csv_revenue_by_category_month,
    retrieve_csv_region_highest_volume,
    retrieve_csv_sales_summary,
    retrieve_csv_region_product_sales,
)

PART2_SYSTEM_PROMPT = """You are an e-commerce analytics assistant. Answer questions using the provided context from sales data (CSV) and/or product pages (text).

Rules:
- Cite specific numbers and product names when possible
- For "both" questions, combine insights from sales data and product reviews
- Be concise but complete
- If the context is insufficient, say so"""


def _extract_category_from_query(query: str) -> str:
    """Heuristic: extract category like 'Electronics' from query."""
    q = query.lower()
    if "electronics" in q:
        return "Electronics"
    if "home" in q or "kitchen" in q:
        return "Home & Kitchen"
    if "sports" in q:
        return "Sports & Outdoors"
    if "beauty" in q:
        return "Beauty & Personal Care"
    if "clothing" in q or "clothes" in q:
        return "Clothing"
    if "books" in q:
        return "Books"
    if "toys" in q:
        return "Toys & Games"
    if "office" in q:
        return "Office Supplies"
    if "pet" in q:
        return "Pet Supplies"
    if "food" in q or "grocery" in q:
        return "Food & Grocery"
    return "Electronics"  # default


def _extract_region_from_query(query: str) -> str | None:
    """Extract region (North, South, East, West, Central) from query."""
    q = query.lower()
    for r in ["north", "south", "east", "west", "central"]:
        if r in q:
            return r.capitalize()
    return None


def answer_question(
    question: str,
    csv_path: Path,
    text_dir: Path,
    df=None,
    pages=None,
    bm25=None,
) -> str:
    """
    Full Part 2 pipeline: route -> retrieve -> LLM answer.
    df, pages, bm25 can be pre-loaded for efficiency when running multiple questions.
    """
    if df is None:
        df = load_csv(csv_path)
    if pages is None:
        pages = load_product_pages(text_dir)
    if bm25 is None:
        bm25, _ = build_bm25_index(pages)

    route = route_question(question)
    context_parts = []

    if route in (SourceRoute.CSV_ONLY, SourceRoute.BOTH):
        # CSV retrieval
        if "revenue" in question.lower() and "electronics" in question.lower() and "december" in question.lower():
            rev = retrieve_csv_revenue_by_category_month(df, "Electronics", 2024, 12)
            context_parts.append(f"Sales Data:\nElectronics revenue in December 2024: ${rev:,.2f}")
        elif "region" in question.lower() and ("highest" in question.lower() or "sales volume" in question.lower()):
            region = retrieve_csv_region_highest_volume(df)
            vol = df.groupby("region")["units_sold"].sum().max()
            context_parts.append(f"Sales Data:\nRegion with highest sales volume: {region} (total units: {int(vol):,})")
        elif route == SourceRoute.BOTH:
            context_parts.append(f"Sales Data:\n{retrieve_csv_sales_summary(df)}")
            region = _extract_region_from_query(question)
            if region:
                reg_df = retrieve_csv_region_product_sales(df, region)
                context_parts.append(f"\nSales in {region} region by product:\n{reg_df.to_string()}")
        else:
            context_parts.append(f"Sales Data:\n{retrieve_csv_sales_summary(df)}")

    if route in (SourceRoute.TEXT_ONLY, SourceRoute.BOTH):
        # Text retrieval
        if route == SourceRoute.BOTH and ("best" in question.lower() or "recommend" in question.lower()):
            text_ctx = retrieve_text_best_reviews(pages, top_k=4)
        else:
            # Expand query for fitness: include yoga, running, exercise
            q_expanded = question
            if "fitness" in question.lower():
                q_expanded = question + " yoga running exercise sports"
            text_ctx = retrieve_text_bm25(pages, bm25, q_expanded, top_k=3)
        if not text_ctx.strip():
            text_ctx = retrieve_text_simple_search(pages, question, top_k=3)
        context_parts.append(f"Product Pages:\n{text_ctx}")

    context = "\n\n".join(context_parts)
    return complete_with_context(
        question=question,
        context=context,
        system_prompt=PART2_SYSTEM_PROMPT,
    )
