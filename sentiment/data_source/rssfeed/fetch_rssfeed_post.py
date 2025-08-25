import feedparser
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Set
import logging
from urllib.parse import urlparse
import time
import re
from enum import Enum

logger = logging.getLogger(__name__)

class FeedCategory(Enum):
    FOREX = "forex"
    CRYPTO = "crypto"
    STOCKS = "stocks"
    GENERAL = "general"
    ECONOMICS = "economics"

class NewsFetcher:
    def __init__(self):
        # Updated list of working RSS feeds
        self.rss_feeds = {
            FeedCategory.FOREX.value: [
                "https://www.fxstreet.com/rss",
                "https://www.investing.com/rss/news.rss",  # More reliable
                "https://finance.yahoo.com/news/topic/forex",
                "https://www.dailyfx.com/feed/rss",  # Updated URL
            ],
            FeedCategory.GENERAL.value: [
                "https://finance.yahoo.com/rss/topstories",
                "https://www.cnbc.com/id/100003114/device/rss/rss.html",
                "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",  # Dow Jones
            ]
        }
        
        # Improved symbol patterns with better specificity
        self.symbol_patterns = {
            "EURUSD": ["EUR/USD", "EUR-USD", "EUR USD", "EURO DOLLAR"],
            "USDJPY": ["USD/JPY", "USD-JPY", "USD JPY", "DOLLAR YEN"],
            "GBPUSD": ["GBP/USD", "GBP-USD", "GBP USD", "CABLE", "STERLING DOLLAR"],
            "USDCHF": ["USD/CHF", "USD-CHF", "USD CHF", "SWISS FRANC"],
            "AUDUSD": ["AUD/USD", "AUD-USD", "AUD USD", "AUSSIE DOLLAR"],
            "USDCAD": ["USD/CAD", "USD-CAD", "USD CAD", "CANADIAN DOLLAR", "LOONIE"],
            "NZDUSD": ["NZD/USD", "NZD-USD", "NZD USD", "KIWI DOLLAR"],
        }
        
        # More specific currency name mappings
        self.currency_names = {
            "EURUSD": ["EURO", "ECB", "EUROPEAN CENTRAL BANK"],
            "USDJPY": ["YEN", "BOJ", "BANK OF JAPAN"],
            "GBPUSD": ["POUND", "STERLING", "BOE", "BANK OF ENGLAND"],
            "USDCHF": ["SWISS FRANC", "SNB", "SWISS NATIONAL BANK"],
            "AUDUSD": ["AUSTRALIAN DOLLAR", "RBA", "RESERVE BANK OF AUSTRALIA"],
            "USDCAD": ["CANADIAN DOLLAR", "BOC", "BANK OF CANADA"],
            "NZDUSD": ["NEW ZEALAND DOLLAR", "KIWI", "RBNZ", "RESERVE BANK OF NEW ZEALAND"],
        }

    def parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse various date formats from RSS feeds with robust error handling"""
        if not date_str or not isinstance(date_str, str):
            return None
            
        try:
            # Try feedparser's built-in parser first
            parsed_time = feedparser._parse_date(date_str)
            if parsed_time:
                # Convert to timezone-aware datetime, then make it naive for comparison
                dt = datetime.fromtimestamp(time.mktime(parsed_time))
                # Make it offset-naive by removing timezone info
                return dt.replace(tzinfo=None)
        except (ValueError, TypeError, AttributeError):
            pass
            
        # Fallback to manual parsing for common formats
        date_formats = [
            "%a, %d %b %Y %H:%M:%S %z",
            "%a, %d %b %Y %H:%M:%S %Z",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%d %H:%M:%S",
            "%d %b %Y %H:%M:%S",
            "%Y-%m-%d",
            "%m/%d/%Y %H:%M:%S",
        ]
        
        for fmt in date_formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                # If it's timezone-aware, make it naive for consistent comparison
                if dt.tzinfo is not None:
                    dt = dt.replace(tzinfo=None)
                return dt
            except ValueError:
                continue
        
        logger.debug(f"Could not parse date: {date_str}")
        return None

    def get_entry_content(self, entry: Dict[str, Any]) -> str:
        """Extract content from RSS entry with multiple fallbacks"""
        content = ""
        
        # Priority order for content extraction
        content_fields = [
            'content', 'summary', 'description', 
            'summary_detail', 'content_detail', 'subtitle'
        ]
        
        for field in content_fields:
            if field in entry:
                if isinstance(entry[field], dict) and 'value' in entry[field]:
                    content = entry[field]['value']
                else:
                    content = str(entry[field])
                if content and content.strip():
                    break
        
        return content.strip() if content else entry.get('title', '')

    def get_host_from_url(self, url: str) -> str:
        """Extract hostname from URL"""
        try:
            netloc = urlparse(url).netloc
            return netloc.replace('www.', '').split(':')[0]
        except:
            return "unknown"

    def contains_symbol(self, text: str, symbol: str) -> bool:
        """Check if text contains the symbol or related terms with fuzzy matching"""
        if not text or not symbol:
            return False
            
        text_upper = text.upper()
        symbol_upper = symbol.upper().replace('/', '')
        
        # Check for exact symbol matches (various formats) with word boundaries
        symbol_patterns = self.symbol_patterns.get(symbol_upper, [symbol_upper])
        
        for pattern in symbol_patterns:
            pattern_upper = pattern.upper()
            # Use word boundaries and require the pattern to be a complete word
            if re.search(rf'\b{re.escape(pattern_upper)}\b', text_upper):
                return True
        
        # Check for currency names and related terms - but be more specific
        currency_terms = self.currency_names.get(symbol_upper, [])
        for term in currency_terms:
            # Only match if it's a complete word and not part of another word
            if re.search(rf'\b{re.escape(term.upper())}\b', text_upper):
                # Additional check to avoid false positives
                if term.upper() in ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"]:
                    # For currency codes, make sure they're not part of a different pair
                    # Don't match "USD" if it's part of "USDCAD" or similar
                    if not re.search(rf'\b{re.escape(term.upper())}[A-Z]{{3}}\b', text_upper):
                        # Also check that it's not part of a different known pair
                        known_pairs = list(self.symbol_patterns.keys())
                        known_pairs.remove(symbol_upper)  # Remove current symbol
                        for other_pair in known_pairs:
                            if other_pair in text_upper:
                                return False
                        return True
                else:
                    # For full names like "EURO", "DOLLAR", etc.
                    return True
                    
        return False

    def fetch_feed(self, feed_url: str, processed_urls: Set[str], symbol: str) -> List[Dict[str, Any]]:
        """Fetch and process a single RSS feed"""
        items = []
        
        try:
            logger.info(f"Parsing feed: {feed_url}")
            feed = feedparser.parse(feed_url)
            
            # Handle redirects
            if hasattr(feed, 'status') and feed.status == 301:
                if hasattr(feed, 'href'):
                    logger.info(f"Following redirect to: {feed.href}")
                    feed = feedparser.parse(feed.href)
            
            # Check feed status
            if hasattr(feed, 'status') and feed.status not in [200, 301]:
                logger.warning(f"Feed {feed_url} returned status: {feed.status}")
                return items
            
            logger.info(f"Found {len(feed.entries)} entries in {feed_url}")
            
            for entry in feed.entries:
                try:
                    # Skip duplicates
                    entry_url = entry.get('link', '')
                    if not entry_url or entry_url in processed_urls:
                        continue
                    processed_urls.add(entry_url)
                    
                    # Parse publication date
                    published_time = self.parse_date(entry.get('published', entry.get('updated', '')))
                    if not published_time:
                        continue
                    
                    # Check if recent (within 15 days) - both should be naive datetimes now
                    cutoff_time = datetime.now() - timedelta(days=15)
                    if published_time < cutoff_time:
                        continue
                    
                    # Get content
                    title = entry.get('title', '')
                    content = self.get_entry_content(entry)
                    full_text = f"{title} {content}"
                    
                    # Check if relevant to our symbol
                    if not self.contains_symbol(full_text, symbol):
                        continue
                    
                    # Create news item
                    news_item = {
                        'source': 'news',
                        'id': f"news_{hash(entry_url) % 1000000:06d}",
                        'title': title,
                        'text': full_text[:800],
                        'url': entry_url,
                        'publisher': self.get_host_from_url(entry_url),
                        'published': published_time.isoformat(),
                        'created_at': datetime.now().isoformat(),
                        'feed_source': feed_url,
                        'symbol': symbol
                    }
                    
                    items.append(news_item)
                    logger.debug(f"Added news item: {title[:60]}...")
                    
                except Exception as e:
                    logger.warning(f"Error processing entry: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error parsing feed {feed_url}: {e}")
        
        return items

    def fetch_news_feeds(self, symbol: str, category: str = "forex") -> List[Dict[str, Any]]:
        """Fetch news from RSS feeds with enhanced filtering and error handling"""
        news_items = []
        processed_urls = set()
        
        logger.info(f"Fetching news for symbol: {symbol}, category: {category}")
        
        try:
            # Get relevant feeds
            feeds = self.rss_feeds.get(category, [])
            if category != "general":
                feeds.extend(self.rss_feeds.get("general", []))
            
            logger.info(f"Checking {len(feeds)} RSS feeds...")
            
            # Process each feed with rate limiting
            for i, feed_url in enumerate(feeds):
                try:
                    items = self.fetch_feed(feed_url, processed_urls, symbol)
                    news_items.extend(items)
                    
                    # Rate limiting between feeds
                    if i < len(feeds) - 1:
                        time.sleep(0.5)
                        
                except Exception as e:
                    logger.error(f"Error processing feed {feed_url}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error in fetch_news_feeds: {e}")
        
        # Sort by publication date (newest first)
        news_items.sort(key=lambda x: x['published'], reverse=True)
        
        logger.info(f"Total news items found for {symbol}: {len(news_items)}")
        return news_items

    def test_feed_connectivity(self):
        """Test which RSS feeds are working"""
        working_feeds = []
        broken_feeds = []
        
        all_feeds = []
        for category in self.rss_feeds.values():
            all_feeds.extend(category)
        
        print("Testing RSS feed connectivity:")
        print("=" * 60)
        
        for feed_url in all_feeds:
            try:
                feed = feedparser.parse(feed_url)
                status = getattr(feed, 'status', 'unknown')
                entries = len(feed.entries)
                
                if status == 200 and entries > 0:
                    working_feeds.append(feed_url)
                    print(f"✅ {feed_url} - {entries} entries")
                else:
                    broken_feeds.append(feed_url)
                    print(f"❌ {feed_url} - Status: {status}, Entries: {entries}")
                    
            except Exception as e:
                broken_feeds.append(feed_url)
                print(f"❌ {feed_url} - Error: {e}")
        
        print("=" * 60)
        print(f"Working feeds: {len(working_feeds)}")
        print(f"Broken feeds: {len(broken_feeds)}")
        
        return working_feeds, broken_feeds

# Singleton instance for easy import
news_fetcher = NewsFetcher()

# Test the fetcher
# if __name__ == "__main__":
#     # Configure logging
#     logging.basicConfig(level=logging.INFO)
    
#     fetcher = NewsFetcher()
    
#     # Test feed connectivity
#     working, broken = fetcher.test_feed_connectivity()
    
#     # Test symbol matching
#     print("\nTesting symbol matching:")
#     test_cases = [
#         ("EURUSD", "EUR/USD reaches new highs", True),
#         ("EURUSD", "USD/CAD analysis", False),  # Should NOT match
#         ("EURUSD", "EURUSD technical analysis", True),
#         ("EURUSD", "Euro dollar forecast", True),
#         ("EURUSD", "USDCAD breaking resistance", False),  # Should NOT match
#     ]
    
#     for symbol, text, expected in test_cases:
#         result = fetcher.contains_symbol(text, symbol)
#         status = "✅" if result == expected else "❌"
#         print(f"{status} {symbol} in '{text}': {result} (expected: {expected})")
    
#     # Test fetching news if we have working feeds
#     if working:
#         print(f"\nTesting news fetch with first working feed: {working[0]}")
#         items = fetcher.fetch_feed(working[0], set(), "EURUSD")
#         print(f"Found {len(items)} relevant items")