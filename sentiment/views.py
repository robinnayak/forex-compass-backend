from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views import View
import json
# from .ai_analyzer import OllamaSentimentAnalyzer, AnalysisAggregator  # Import the new AI classes
from sentiment.ai.ai_analyze import OllamaSentimentAnalyzer, AnalysisAggregator
import requests
import os
from dotenv import load_dotenv
from sentiment.data_source.reddit.fetch_reddit_post import fetch_reddit_posts
from sentiment.data_source.rssfeed.fetch_rssfeed_post import NewsFetcher
from sentiment.utils import load_symbol_metadata, standardize_currency_pair   
import logging
from datetime import datetime

# Set up logging
logger = logging.getLogger(__name__)

load_dotenv()

# Configuration
TEST_CONFIG = {
    "reddit_credentials": {
        "client_id": os.getenv("REDDIT_CLIENT_ID", ""),
        "client_secret": os.getenv("REDDIT_CLIENT_SECRET", ""),
        "user_agent": os.getenv("REDDIT_USER_AGENT", "sentiment-python/1.0")
    },
    "reddit_max_items": int(os.getenv("REDDIT_MAX_ITEMS", 50)),
    "reddit_min_upvotes": int(os.getenv("REDDIT_MIN_UPVOTES", 2)),
    "reddit_min_comments": int(os.getenv("REDDIT_MIN_COMMENTS", 1))
}

# Initialize AI components
ai_analyzer = OllamaSentimentAnalyzer(
    model=os.getenv("OLLAMA_MODEL", "gemma3:latest"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    timeout=int(os.getenv("OLLAMA_TIMEOUT", 300))
)
analysis_aggregator = AnalysisAggregator()

def welcome_screen(request):
    return JsonResponse({"message": "Welcome to the Forex Compass API!"})

class SentimentAnalysisView(View):
    @csrf_exempt
    def dispatch(self, *args, **kwargs):
        return super().dispatch(*args, **kwargs)
    
    def get(self, request, symbol=None):
        # Use symbol from URL
        if not symbol:
            return JsonResponse({'error': 'Symbol is required.'}, status=400)
        return JsonResponse({
            "message": f"Send a POST request to analyze sentiment for symbol: {symbol}",
            "available_parameters": {
                "symbol": "Currency pair (e.g., EURUSD, GBPUSD)",
                "include_reddit": "boolean (default: true)",
                "include_news": "boolean (default: true)",
                "analyze_sentiment": "boolean (default: true)",
                "ai_model": "string (default: gemma3:latest)",
                "max_items": "integer (default: 20 per source)"
            },
            "supported_symbols": ["EURUSD", "USDJPY", "GBPUSD", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"],
            "symbol": symbol
        })

    def post(self, request, symbol=None):
        try:
            # Parse JSON data
            try:
                data = json.loads(request.body)
            except json.JSONDecodeError:
                data = {}
            # Use symbol from URL if not in body
            if not symbol:
                symbol = data.get('symbol')
            if not symbol:
                return JsonResponse({'error': 'Symbol is required.'}, status=400)
            # Get optional parameters
            include_reddit = data.get('include_reddit', True)
            include_news = data.get('include_news', True)
            analyze_sentiment = data.get('analyze_sentiment', True)
            max_items = data.get('max_items', 20)
            # Standardize the symbol
            standardized_symbol = standardize_currency_pair(symbol)
            if not standardized_symbol:
                return JsonResponse({'error': f'Invalid currency pair: {symbol}'}, status=400)
            # Load metadata
            metadata = load_symbol_metadata(standardized_symbol.replace("/", ""))
            if not metadata:
                metadata = {
                    "keywords": [standardized_symbol],
                    "subs": ["Forex", "investing", "economics"],
                    "category": "forex"
                }
            logger.info(f"Processing symbol: {standardized_symbol}")
            logger.info(f"Include Reddit: {include_reddit}, Include News: {include_news}, Analyze Sentiment: {analyze_sentiment}")
            reddit_data = []
            news_data = []
            analyses = []
            # Fetch Reddit posts if enabled
            if include_reddit:
                try:
                    reddit_data = fetch_reddit_posts(standardized_symbol, metadata)
                    if isinstance(reddit_data, dict) and 'error' in reddit_data:
                        logger.warning(f"Reddit fetch error: {reddit_data['error']}")
                        reddit_data = []
                    else:
                        reddit_data = reddit_data[:max_items]
                        logger.info(f"Reddit posts found: {len(reddit_data)}")
                        if analyze_sentiment and reddit_data:
                            reddit_analysis = ai_analyzer.analyze_sentiment(
                                reddit_data, standardized_symbol, "reddit"
                            )
                            reddit_analysis['source_type'] = 'reddit'
                            analyses.append(reddit_analysis)
                except Exception as e:
                    logger.error(f"Error fetching Reddit posts: {e}")
                    reddit_data = []
            # Fetch news items if enabled
            if include_news:
                try:
                    category = metadata.get('category', 'forex')
                    news_fetcher = NewsFetcher()
                    news_data = news_fetcher.fetch_news_feeds(standardized_symbol, category)
                    news_data = news_data[:max_items]
                    logger.info(f"News items found: {len(news_data)}")
                    if analyze_sentiment and news_data:
                        news_analysis = ai_analyzer.analyze_sentiment(
                            news_data, standardized_symbol, "news"
                        )
                        news_analysis['source_type'] = 'news'
                        analyses.append(news_analysis)
                except Exception as e:
                    logger.error(f"Error fetching news: {e}")
                    news_data = []
            # Prepare response data
            response_data = {
                "symbol": standardized_symbol,
                "metadata": metadata,
                "sources": {
                    "reddit": {
                        "enabled": include_reddit,
                        "items_found": len(reddit_data),
                        "items": reddit_data[:10]  # Return only first 10 items for response
                    },
                    "news": {
                        "enabled": include_news,
                        "items_found": len(news_data),
                        "items": news_data[:10]  # Return only first 10 items for response
                    }
                },
                "total_items": len(reddit_data) + len(news_data),
                "timestamp": datetime.now().isoformat()
            }
            
            # Perform sentiment analysis aggregation if requested
            if analyze_sentiment:
                try:
                    if analyses:
                        final_analysis = analysis_aggregator.aggregate_analyses(analyses)
                        response_data["sentiment_analysis"] = final_analysis
                        logger.info("Sentiment analysis completed successfully")
                    else:
                        response_data["sentiment_analysis"] = {
                            "error": "No data available for analysis",
                            "symbol": standardized_symbol,
                            "status": "no_data"
                        }
                except Exception as e:
                    logger.error(f"Error in sentiment analysis: {e}")
                    response_data["sentiment_analysis"] = {
                        "error": str(e),
                        "symbol": standardized_symbol,
                        "status": "analysis_failed"
                    }
            
            return JsonResponse(response_data, safe=False)
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis view: {e}")
            return JsonResponse({
                'error': 'Internal server error',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }, status=500)

class HealthCheckView(View):
    def get(self, request):
        # Test Ollama connection
        ollama_status = "unknown"
        try:
            # Simple test to check if Ollama is reachable
            response = requests.get(f"{os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}/api/tags", timeout=5)
            ollama_status = "connected" if response.status_code == 200 else "unavailable"
        except:
            ollama_status = "unavailable"
        
        return JsonResponse({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "reddit_configured": bool(os.getenv("REDDIT_CLIENT_ID")),
                "ollama_configured": ollama_status,
                "version": "1.1.0"
            },
            "ollama_model": os.getenv("OLLAMA_MODEL", "gemma3:latest")
        })

class RedditAnalysisView(View):
    @csrf_exempt
    def dispatch(self, *args, **kwargs):
        return super().dispatch(*args, **kwargs)
    
    def post(self, request):
        try:
            data = json.loads(request.body)
            symbol = data.get('symbol')
            analyze_sentiment = data.get('analyze_sentiment', False)
            max_items = data.get('max_items', 20)
            
            if not symbol:
                return JsonResponse({'error': 'Symbol is required.'}, status=400)
            
            standardized_symbol = standardize_currency_pair(symbol)
            if not standardized_symbol:
                return JsonResponse({'error': f'Invalid currency pair: {symbol}'}, status=400)
            
            metadata = load_symbol_metadata(standardized_symbol.replace("/", ""))
            if not metadata:
                metadata = {
                    "keywords": [standardized_symbol],
                    "subs": ["Forex", "investing", "economics"],
                    "category": "forex"
                }
            
            reddit_data = fetch_reddit_posts(standardized_symbol, metadata)
            
            if isinstance(reddit_data, dict) and 'error' in reddit_data:
                return JsonResponse(reddit_data, status=400)
            
            # Limit items for response
            response_data = reddit_data[:max_items]
            
            result = {
                "symbol": standardized_symbol,
                "items_found": len(reddit_data),
                "items": response_data,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add sentiment analysis if requested
            if analyze_sentiment and reddit_data:
                analysis = ai_analyzer.analyze_sentiment(
                    reddit_data[:10], standardized_symbol, "reddit"
                )
                result["sentiment_analysis"] = analysis
            
            return JsonResponse(result, safe=False)
            
        except Exception as e:
            return JsonResponse({
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }, status=500)

class NewsAnalysisView(View):
    @csrf_exempt
    def dispatch(self, *args, **kwargs):
        return super().dispatch(*args, **kwargs)
    
    def post(self, request):
        try:
            data = json.loads(request.body)
            symbol = data.get('symbol')
            analyze_sentiment = data.get('analyze_sentiment', False)
            max_items = data.get('max_items', 20)
            
            if not symbol:
                return JsonResponse({'error': 'Symbol is required.'}, status=400)
            
            standardized_symbol = standardize_currency_pair(symbol)
            if not standardized_symbol:
                return JsonResponse({'error': f'Invalid currency pair: {symbol}'}, status=400)
            
            category = data.get('category', 'forex')
            news_fetcher = NewsFetcher()
            news_data = news_fetcher.fetch_news_feeds(standardized_symbol, category)

            # Limit items for response
            response_data = news_data[:max_items]
            
            result = {
                "symbol": standardized_symbol,
                "category": category,
                "items_found": len(news_data),
                "items": response_data,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add sentiment analysis if requested
            if analyze_sentiment and news_data:
                analysis = ai_analyzer.analyze_sentiment(
                    news_data[:10], standardized_symbol, "news"
                )
                result["sentiment_analysis"] = analysis
            
            return JsonResponse(result, safe=False)
            
        except Exception as e:
            return JsonResponse({
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }, status=500)

class AIAnalysisView(View):
    """Dedicated endpoint for AI sentiment analysis only"""
    @csrf_exempt
    def dispatch(self, *args, **kwargs):
        return super().dispatch(*args, **kwargs)
    
    def post(self, request):
        try:
            data = json.loads(request.body)
            items = data.get('items', [])
            symbol = data.get('symbol')
            source = data.get('source', 'unknown')
            
            if not items:
                return JsonResponse({'error': 'No items provided for analysis'}, status=400)
            if not symbol:
                return JsonResponse({'error': 'Symbol is required'}, status=400)
            
            # Standardize the symbol
            standardized_symbol = standardize_currency_pair(symbol)
            if not standardized_symbol:
                return JsonResponse({'error': f'Invalid currency pair: {symbol}'}, status=400)
            
            # Perform AI analysis
            analysis = ai_analyzer.analyze_sentiment(
                items, standardized_symbol, source
            )
            
            return JsonResponse({
                "symbol": standardized_symbol,
                "source": source,
                "analysis": analysis,
                "items_analyzed": len(items),
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            return JsonResponse({
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }, status=500)

# URL configuration example (add to your urls.py)
"""
from django.urls import path
from . import views

urlpatterns = [
    path('', views.welcome_screen, name='welcome'),
    path('analyze/', views.SentimentAnalysisView.as_view(), name='analyze'),
    path('health/', views.HealthCheckView.as_view(), name='health'),
    path('analyze/reddit/', views.RedditAnalysisView.as_view(), name='analyze_reddit'),
    path('analyze/news/', views.NewsAnalysisView.as_view(), name='analyze_news'),
    path('analyze/ai/', views.AIAnalysisView.as_view(), name='analyze_ai'),
]
"""