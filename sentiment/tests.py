from django.test import TestCase
from sentiment.data_source.rssfeed.fetch_rssfeed_post import NewsFetcher, FeedCategory
import feedparser
from datetime import datetime, timedelta
import logging

# Set up logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsFetcherTests(TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.fetcher = NewsFetcher()
        self.test_symbol = "EURUSD"
        self.test_category = "forex"
        
    def test_news_fetcher_initialization(self):
        """Test that NewsFetcher initializes correctly"""
        self.assertIsInstance(self.fetcher, NewsFetcher)
        self.assertIsNotNone(self.fetcher.rss_feeds)
        self.assertIsNotNone(self.fetcher.symbol_patterns)
        self.assertIsNotNone(self.fetcher.currency_names)
        
    def test_symbol_patterns_exist(self):
        """Test that symbol patterns are defined for major currency pairs"""
        major_pairs = ["EURUSD", "USDJPY", "GBPUSD"]
        
        for pair in major_pairs:
            with self.subTest(pair=pair):
                self.assertIn(pair, self.fetcher.symbol_patterns)
                patterns = self.fetcher.symbol_patterns[pair]
                self.assertGreater(len(patterns), 0, f"No patterns found for {pair}")

    def test_parse_date_valid_formats(self):
        """Test date parsing with various valid formats"""
        test_cases = [
            ("Mon, 23 Aug 2025 10:00:00 GMT", datetime(2025, 8, 23, 10, 0, 0)),
            ("2025-08-23T10:00:00+00:00", datetime(2025, 8, 23, 10, 0, 0)),
        ]
        
        for date_str, expected in test_cases:
            with self.subTest(date_str=date_str):
                result = self.fetcher.parse_date(date_str)
                self.assertIsNotNone(result, f"Failed to parse: {date_str}")
                self.assertEqual(result.year, expected.year)
                self.assertEqual(result.month, expected.month)

    def test_contains_symbol_positive_cases(self):
        """Test symbol matching with positive cases"""
        test_cases = [
            ("EURUSD", "EUR/USD reaches new highs"),
            ("EURUSD", "EUR-USD technical analysis"),
            ("USDJPY", "USD/JPY remains stable"),
        ]
        
        for symbol, text in test_cases:
            with self.subTest(symbol=symbol, text=text):
                result = self.fetcher.contains_symbol(text, symbol)
                self.assertTrue(result, f"Should match {symbol} in: {text}")

    def test_contains_symbol_negative_cases(self):
        """Test symbol matching with negative cases"""
        test_cases = [
            ("EURUSD", "Random news without currency mention"),
            ("EURUSD", "USD/CAD analysis"),  # Different pair - should NOT match
            ("EURUSD", "EUR mentioned alone"),  # Should not match without context
            ("EURUSD", "USD mentioned alone"),  # Should not match without context
            ("EURUSD", "USDCAD technical analysis"),  # Different pair
        ]
        
        for symbol, text in test_cases:
            with self.subTest(symbol=symbol, text=text):
                result = self.fetcher.contains_symbol(text, symbol)
            self.assertFalse(result, f"Should not match {symbol} in: {text}")

    def test_get_entry_content(self):
        """Test content extraction from entry"""
        test_entry = {
            'title': 'Test Title',
            'summary': 'Test summary content'
        }
        
        result = self.fetcher.get_entry_content(test_entry)
        self.assertEqual(result, "Test summary content")

    def test_get_host_from_url(self):
        """Test URL hostname extraction"""
        test_cases = [
            ("https://www.example.com/path", "example.com"),
            ("http://subdomain.example.com:8080/path", "subdomain.example.com"),
        ]
        
        for url, expected in test_cases:
            with self.subTest(url=url):
                result = self.fetcher.get_host_from_url(url)
                self.assertEqual(result, expected)

    def test_feed_parsing_integration(self):
        """Test integration with actual feed parsing"""
        result = self.fetcher.fetch_news_feeds(self.test_symbol, self.test_category)
        self.assertIsInstance(result, list, "Should return a list even if empty")

# Simple test function to check feeds
def test_feed_connectivity():
    """Test if RSS feeds are accessible"""
    fetcher = NewsFetcher()
    
    test_feeds = [
        "https://www.forexlive.com/feed/",
        "https://www.fxstreet.com/rss",
        "https://www.reuters.com/rss/finance/forex",
    ]
    
    print("Testing RSS feed connectivity:")
    print("=" * 50)
    
    for feed_url in test_feeds:
        try:
            feed = feedparser.parse(feed_url)
            status = getattr(feed, 'status', 'unknown')
            entries = len(feed.entries)
            
            print(f"📰 {feed_url}")
            print(f"   Status: {status}")
            print(f"   Entries: {entries}")
            
            if entries > 0:
                print(f"   Sample: {feed.entries[0].get('title', 'No title')[:60]}...")
            
            print()
            
        except Exception as e:
            print(f"❌ {feed_url}")
            print(f"   Error: {e}")
            print()

if __name__ == "__main__":
    test_feed_connectivity()