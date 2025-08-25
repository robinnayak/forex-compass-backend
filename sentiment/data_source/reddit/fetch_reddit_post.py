from dotenv import load_dotenv
import praw
import logging
import os 
from datetime import datetime
from typing import Dict, Any, List, Optional
import time
import requests
from requests.exceptions import Timeout

load_dotenv()  # Load environment variables

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RedditFetcher:
    """Class to handle Reddit API operations and post fetching"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.reddit_client = None
        self.initialize_reddit_client()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration with environment variables"""
        return {
            "reddit_credentials": {
                "client_id": os.getenv("REDDIT_CLIENT_ID", ""),
                "client_secret": os.getenv("REDDIT_CLIENT_SECRET", ""),
                "user_agent": os.getenv("REDDIT_USER_AGENT", "sentiment-python/1.0"),
            },
            "reddit_max_items": int(os.getenv("REDDIT_MAX_ITEMS", 50)),
            "reddit_min_upvotes": int(os.getenv("REDDIT_MIN_UPVOTES", 2)),
            "reddit_min_comments": int(os.getenv("REDDIT_MIN_COMMENTS", 1)),
            "request_timeout": int(os.getenv("REDDIT_TIMEOUT", 30)),
            "rate_limit_delay": float(os.getenv("REDDIT_RATE_LIMIT_DELAY", 2.0)),
        }
    
    def initialize_reddit_client(self) -> bool:
        """Initialize and authenticate Reddit client"""
        try:
            if not self.config.get("reddit_credentials"):
                logger.error("No reddit_credentials in config")
                return False
            
            creds = self.config["reddit_credentials"]
            
            if not creds["client_id"] or not creds["client_secret"]:
                logger.error("Missing Reddit credentials")
                return False
            
            self.reddit_client = praw.Reddit(
                client_id=creds["client_id"],
                client_secret=creds["client_secret"],
                user_agent=creds["user_agent"],
                timeout=self.config["request_timeout"]
            )
            
            # Test authentication
            try:
                user = self.reddit_client.user.me()
                logger.info(f"Authenticated as: {user}")
                return True
            except Exception as auth_error:
                logger.error(f"Authentication failed: {auth_error}")
                self.reddit_client = None
                return False
                
        except Exception as e:
            logger.error(f"Error initializing Reddit client: {e}")
            self.reddit_client = None
            return False
    
    def is_authenticated(self) -> bool:
        """Check if Reddit client is authenticated"""
        return self.reddit_client is not None
    
    def search_subreddit(self, subreddit_name: str, query: str, limit: int = 50, 
                        time_filter: str = "month") -> List[Any]:
        """Search for posts in a specific subreddit"""
        try:
            subreddit = self.reddit_client.subreddit(subreddit_name)
            return list(subreddit.search(
                query=query,
                limit=limit,
                time_filter=time_filter,
            ))
        except Exception as e:
            logger.error(f"Error searching subreddit {subreddit_name}: {e}")
            return []
    
    def process_post(self, post) -> Optional[Dict[str, Any]]:
        """Process a single Reddit post and extract relevant information"""
        try:
            # Apply minimum filters
            if (post.score >= self.config["reddit_min_upvotes"] and
                post.num_comments >= self.config["reddit_min_comments"]):
                
                full_text = f"{post.title} {post.selftext}"
                return {
                    "id": post.id,
                    "title": post.title,
                    "text": full_text[:500] + "..." if len(full_text) > 500 else full_text,
                    "url": f"https://reddit.com{post.permalink}",
                    "subreddit": post.subreddit.display_name,
                    "upvotes": post.score,
                    "comments": post.num_comments,
                    "upvote_ratio": post.upvote_ratio,
                    "created_utc": post.created_utc,
                    "created_at": datetime.fromtimestamp(post.created_utc).isoformat(),
                    "author": str(post.author) if post.author else "Unknown",
                    "nsfw": post.over_18,
                }
        except Exception as e:
            logger.warning(f"Error processing post {getattr(post, 'id', 'unknown')}: {e}")
        
        return None
    
    def fetch_posts(self, symbol: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fetch relevant posts from Reddit for a given symbol"""
        print(f"Fetching Reddit posts for symbol: {symbol}")
        
        if not self.is_authenticated():
            error_msg = "Reddit client not authenticated. Check your credentials."
            logger.error(error_msg)
            return []
        
        posts = []
        
        try:
            # Format symbol for search
            search_symbol = symbol.replace('/', ' ') if '/' in symbol else symbol
            
            # Get subreddits and keywords from metadata
            subreddits = metadata.get("subs", ["Forex", "Crypto", "Investing"])
            keywords = metadata.get("keywords", []) + [
                "Dollar", "Fed", "FOMC", "Federal Reserve", "Powell", "Lagarde",
                "European Central Bank", "Eurozone", "inflation", "CPI", "GDP",
                "rate hike", "rate cut", "interest rates", "forex", "FX", "currency pair",
            ]

            search_query = f"{search_symbol} {' '.join(keywords[:2])}"
            print(f"Search query: {search_query}")
            print(f"Searching in subreddits: {subreddits}")
            
            # Search each subreddit
            for i, subreddit_name in enumerate(subreddits):
                try:
                    print(f"Searching in subreddit: {subreddit_name}")
                    
                    # Rate limiting between subreddit searches
                    if i > 0:
                        time.sleep(self.config["rate_limit_delay"])
                    
                    # Search for posts
                    search_results = self.search_subreddit(
                        subreddit_name=subreddit_name,
                        query=search_query,
                        limit=self.config["reddit_max_items"],
                        time_filter="month"
                    )
                    
                    # Process each post
                    for post in search_results:
                        processed_post = self.process_post(post)
                        if processed_post:
                            posts.append(processed_post)
                            
                    print(f"Found {len([p for p in search_results])} posts in {subreddit_name}, kept {len([p for p in posts if p['subreddit'] == subreddit_name])}")
                    
                except Exception as e:
                    logger.error(f"Error processing subreddit {subreddit_name}: {e}")
                    continue
                
        except Exception as e:
            logger.error(f"Error in fetch_posts: {e}")
        
        # Sort by date (newest first)
        posts.sort(key=lambda x: x['created_utc'], reverse=True)
        print(f"Total posts found for {symbol}: {len(posts)}")
        
        return posts
    
    def get_subreddit_info(self, subreddit_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a subreddit"""
        if not self.is_authenticated():
            return None
        
        try:
            subreddit = self.reddit_client.subreddit(subreddit_name)
            return {
                "name": subreddit.display_name,
                "subscribers": subreddit.subscribers,
                "description": subreddit.description,
                "created_utc": subreddit.created_utc,
                "nsfw": subreddit.over18,
            }
        except Exception as e:
            logger.error(f"Error getting subreddit info for {subreddit_name}: {e}")
            return None
    
    def test_connection(self) -> Dict[str, Any]:
        """Test Reddit connection and return status"""
        status = {
            "authenticated": self.is_authenticated(),
            "client_initialized": self.reddit_client is not None,
            "config_loaded": bool(self.config),
        }
        
        if self.is_authenticated():
            try:
                user = self.reddit_client.user.me()
                status["user"] = str(user)
                status["success"] = True
            except Exception as e:
                status["error"] = str(e)
                status["success"] = False
        else:
            status["success"] = False
            status["error"] = "Not authenticated"
        
        return status


# Singleton instance for easy import
reddit_fetcher = RedditFetcher()

# Backward compatibility functions
def check_credential(test_config=None):
    """Backward compatibility function"""
    fetcher = RedditFetcher(test_config)
    return fetcher.reddit_client if fetcher.is_authenticated() else False

def fetch_reddit_posts(symbol, metadata):
    """Backward compatibility function"""
    return reddit_fetcher.fetch_posts(symbol, metadata)


# Example usage and testing
if __name__ == "__main__":
    # Test the Reddit fetcher
    print("Testing RedditFetcher...")
    
    # Test connection
    status = reddit_fetcher.test_connection()
    print(f"Connection status: {status}")
    
    if status["authenticated"]:
        # Test with sample metadata
        sample_metadata = {
            "subs": ["Forex", "investing"],
            "keywords": ["EUR/USD", "Euro Dollar"],
            "category": "forex"
        }
        
        # Fetch posts
        posts = reddit_fetcher.fetch_posts("EURUSD", sample_metadata)
        print(f"Found {len(posts)} posts")
        
        # Display first few posts
        for i, post in enumerate(posts[:3]):
            print(f"\nPost {i+1}:")
            print(f"  Title: {post['title'][:50]}...")
            print(f"  Subreddit: {post['subreddit']}")
            print(f"  Upvotes: {post['upvotes']}")
            print(f"  Date: {post['created_at'][:10]}")
    else:
        print("Cannot test without valid Reddit credentials")