import os
import unittest
from storage.database import Database
from scraper.arxiv_scraper import ArxivScraper
from analyzer.local_evaluator import LocalPaperEvaluator
from search_papers import search_and_evaluate

class TestSearchKeyword(unittest.TestCase):
    def test_arxiv_relevance_search(self):
        scraper = ArxivScraper()
        papers = scraper.search("statistical arbitrage", max_results=5)
        self.assertGreater(len(papers), 0, "Expected at least 1 paper from arXiv search")
        
        stat_arb_hits = 0
        for p in papers:
            combined = f"{p['title']} {p['abstract']}".lower()
            if "statistical arbitrage" in combined or "arbitrage" in combined or "pairs trading" in combined:
                stat_arb_hits += 1
                
        self.assertGreaterEqual(stat_arb_hits, len(papers) * 0.8, "At least 80% of papers should be directly relevant to statistical arbitrage")

    def test_local_evaluator_discrimination(self):
        evaluator = LocalPaperEvaluator()
        
        # 1. High-quality statistical arbitrage paper
        good_paper = {
            "title": "Deep Learning Statistical Arbitrage",
            "abstract": "We develop a deep learning framework for statistical arbitrage in the US equity market. Using cointegration and neural networks, we optimize pairs trading portfolios, achieving Sharpe ratios exceeding 2.5 with controlled drawdowns after accounting for transaction costs and slippage.",
            "category": "q-fin.PM",
            "citations": 45,
            "published_date": "2024-01-15"
        }
        
        # 2. General non-arbitrage econometrics paper
        mediocre_paper = {
            "title": "Testing for Heterogeneous Treatment Effects in Regression Discontinuity Designs",
            "abstract": "We develop tests for treatment effect heterogeneity in regression discontinuity designs using local polynomial estimators.",
            "category": "econ.EM",
            "citations": 2,
            "published_date": "2026-09-23"
        }
        
        # 3. Off-topic paper
        bad_paper = {
            "title": "Credit Card Fraud Detection Using State-of-the-Art Machine Learning",
            "abstract": "Credit card fraud detection is a critical issue in banking. We benchmark random forest models on unbalanced transaction data.",
            "category": "Computer Science",
            "citations": 10,
            "published_date": "2023-01-10"
        }
        
        res_good = evaluator.evaluate(good_paper, query="statistical arbitrage")
        res_med = evaluator.evaluate(mediocre_paper, query="statistical arbitrage")
        res_bad = evaluator.evaluate(bad_paper, query="statistical arbitrage")
        
        self.assertGreaterEqual(res_good["score"], 70, f"Expected high score for stat-arb paper, got {res_good['score']}")
        self.assertTrue(res_good["is_quant_finance"])
        self.assertGreater(len(res_good["hook"]), 10)
        
        self.assertLess(res_med["score"], res_good["score"], "General paper should score lower than topic-matching paper")
        self.assertLessEqual(res_bad["score"], 20, f"Off-topic paper should score <= 20, got {res_bad['score']}")
        self.assertFalse(res_bad["is_quant_finance"])

    def test_database_download_status(self):
        db = Database("data/papers.db")
        status_new = db.get_download_status("arxiv:9999.99999")
        self.assertEqual(status_new["status"], "new")
        self.assertIsNone(status_new["paper_id"])
        self.assertFalse(db.is_pdf_downloaded("arxiv:9999.99999"))

    def test_search_and_evaluate_function(self):
        results = search_and_evaluate("statistical arbitrage", limit=2, download_pdfs=False)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        for p in results:
            self.assertIn("score", p)
            self.assertIn("hook", p)
            self.assertIn("is_downloaded", p)

if __name__ == "__main__":
    unittest.main()