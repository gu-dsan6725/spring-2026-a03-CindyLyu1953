"""Test cases for Part 2: Query Router."""

import pytest

from src.part2_router import route_question, SourceRoute


class TestRouteQuestion:
    """Test query routing to csv_only, text_only, or both."""

    def test_csv_only_revenue_electronics_december(self):
        q = "What was the total revenue for Electronics category in December 2024?"
        assert route_question(q) == SourceRoute.CSV_ONLY

    def test_csv_only_region_highest_volume(self):
        q = "Which region had the highest sales volume?"
        assert route_question(q) == SourceRoute.CSV_ONLY

    def test_text_only_key_features(self):
        q = "What are the key features of the Wireless Bluetooth Headphones?"
        assert route_question(q) == SourceRoute.TEXT_ONLY

    def test_text_only_customers_say_cleaning(self):
        q = "What do customers say about the Air Fryer's ease of cleaning?"
        assert route_question(q) == SourceRoute.TEXT_ONLY

    def test_both_best_reviews_selling(self):
        q = "Which product has the best customer reviews and how well is it selling?"
        assert route_question(q) == SourceRoute.BOTH

    def test_both_recommend_fitness_west(self):
        q = "I want a product for fitness that is highly rated and sells well in the West region. What do you recommend?"
        assert route_question(q) == SourceRoute.BOTH
