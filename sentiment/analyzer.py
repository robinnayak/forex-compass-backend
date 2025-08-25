import requests
import json
import re
from datetime import datetime, timedelta
import pandas as pd
import praw
import feedparser
from typing import List, Dict, Any, Optional
import logging
from bs4 import BeautifulSoup
import time
import numpy as np
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TradingSentimentAnalyzer:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        print("Initializing TradingSentimentAnalyzer with config:")
        print(json.dumps(config, indent=2))

        # Initialize Reddit client if credentials are provided
        self.reddit = None
        if config.get('reddit_credentials'):
            try:
                self.reddit = praw.Reddit(
                    client_id=config['reddit_credentials']['client_id'],
                    client_secret=config['reddit_credentials']['client_secret'],
                    user_agent=config['reddit_credentials']['user_agent']
                )
                # Test the connection
                self.reddit.user.me()
                print("Reddit API connection successful")
            except Exception as e:
                logger.error(f"Failed to initialize Reddit client: {e}")

    def calculate_metrics(self, items: List[Dict[str, Any]], source: str) -> List[Dict[str, Any]]:
        """Calculate engagement, trust, relevance, and sentiment metrics"""
        keywords = ["crypto", "btc", "eth", "forex", "scalping", "technical analysis", 
                   "trading", "market", "price", "analysis", "signal"]
        processed_items = []
        
        for item in items:
            text = item.get('text', '').lower()
            title = item.get('title', '').lower()
            full_text = f"{title} {text}"
            
            # Engagement score for different sources
            if source == 'reddit':
                engagement_score = round((item.get('upvotes', 0) * item.get('upvote_ratio', 0.5)))
                subscribers = item.get('subreddit_subscribers', 0)
                trust_score = round(np.log10(subscribers + 1), 2) if subscribers else 0.5
            else:  # news
                engagement_score = 1  # Base engagement for news
                trust_score = 1.0  # Base trust for news sources
            
            # Relevance score: keyword match count
            relevance_score = sum(1 for kw in keywords if kw in full_text)
            
            # Simple sentiment analysis
            sentiment_score = 0.0
            positive_terms = ["bull", "positive", "up", "buy", "long", "strong", "growth", "rally"]
            negative_terms = ["bear", "negative", "down", "sell", "short", "weak", "drop", "crash"]
            
            for term in positive_terms:
                if term in full_text:
                    sentiment_score += 0.3
            
            for term in negative_terms:
                if term in full_text:
                    sentiment_score -= 0.3
            
            # Normalize sentiment score
            sentiment_score = max(-1.0, min(1.0, sentiment_score))
            
            # Add metrics to the item
            item['metrics'] = {
                'engagement_score': engagement_score,
                'trust_score': trust_score,
                'relevance_score': relevance_score,
                'sentiment_score': round(sentiment_score, 3),
                'source_type': source
            }
            
            processed_items.append(item)
        
        return processed_items

    def analyze_sentiment_with_ai(self, items: List[Dict[str, Any]], symbol: str, source: str):
        """
        Analyze sentiment using Ollama model running locally.
        """
        prompt = self._create_ai_prompt(items, symbol, source)
        ollama_url = "http://localhost:11434/api/generate"
        payload = {
            "model": "gemma3:latest",
            "prompt": prompt,
            "stream": False
        }
        
        try:
            response = requests.post(ollama_url, json=payload, timeout=300)
            response.raise_for_status()
            result = response.json()
            
            ai_output = result.get("response", "")
            
            # Try to parse JSON from the AI output
            try:
                json_match = re.search(r'```json\s*(.*?)\s*```', ai_output, re.DOTALL)
                if json_match:
                    ai_output = json_match.group(1)
                
                analysis = json.loads(ai_output.strip())
                return analysis
            except json.JSONDecodeError:
                try:
                    json_str = re.search(r'\{.*\}', ai_output, re.DOTALL)
                    if json_str:
                        analysis = json.loads(json_str.group())
                        return analysis
                    else:
                        return {
                            "symbol": symbol,
                            "signal": "Neutral",
                            "rationale": "AI response could not be parsed as JSON",
                            "ai_output": ai_output,
                            "date": datetime.now().strftime("%Y-%m-%d")
                        }
                except Exception as e:
                    logger.error(f"Failed to parse AI response: {e}")
                    return {
                        "symbol": symbol,
                        "signal": "Neutral",
                        "rationale": f"Error parsing AI response: {str(e)}",
                        "raw_output": ai_output,
                        "date": datetime.now().strftime("%Y-%m-%d")
                    }
                    
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API request error: {e}")
            return {
                "symbol": symbol,
                "signal": "Neutral",
                "rationale": "Ollama API unavailable",
                "error": str(e),
                "date": datetime.now().strftime("%Y-%m-%d")
            }
        except Exception as e:
            logger.error(f"Unexpected error in AI analysis: {e}")
            return {
                "symbol": symbol,
                "signal": "Neutral",
                "rationale": "Unexpected error in analysis",
                "error": str(e),
                "date": datetime.now().strftime("%Y-%m-%d")
            }
    
    def _create_ai_prompt(self, items: List[Dict[str, Any]], symbol: str, source: str) -> str:
        """Create a prompt for AI analysis"""
        examples = []
        for item in items[:8]:  # Limit to 8 examples
            examples.append({
                "title": item.get('title', ''),
                "text": (item.get('text', '')[:300] + "...") if len(item.get('text', '')) > 300 else item.get('text', ''),
                "source": item.get('publisher') or item.get('subreddit', ''),
                "createdAt": item.get('created_at', ''),
                "metrics": item.get('metrics', {})
            })
        
        prompt = f"""
        You are a financial analyst specializing in currency trading. Analyze the following {source} content about {symbol} and provide a trading recommendation.
        
        Content to analyze:
        {json.dumps(examples, indent=2)}
        
        Provide your analysis in JSON format with the following structure:
        {{
          "symbol": "{symbol}",
          "signal": "Buy|Sell|Neutral",
          "impact": "Low|Medium|High",
          "confidence": "XX%",
          "rationale": "Concise explanation based on sentiment analysis",
          "key_insights": ["insight1", "insight2"],
          "sources_analyzed": {len(items)},
          "timestamp": "{datetime.now().isoformat()}"
        }}
        
        Important: Return ONLY valid JSON, without any additional text or explanation.
        """
        
        return prompt
    
    def aggregate_analyses(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate multiple analyses into a final recommendation"""
        if not analyses:
            return {
                "symbol": "Unknown",
                "signal": "Neutral",
                "confidence": "50%",
                "rationale": "No data available for analysis",
                "sources_analyzed": 0,
                "timestamp": datetime.now().isoformat()
            }
        
        # Filter out analyses that don't have proper signals
        valid_analyses = [a for a in analyses if a.get('signal') in ['Buy', 'Sell', 'Neutral']]
        
        if not valid_analyses:
            return {
                "symbol": analyses[0].get('symbol', 'Unknown') if analyses else "Unknown",
                "signal": "Neutral",
                "confidence": "50%",
                "rationale": "No valid analyses available",
                "sources_analyzed": 0,
                "timestamp": datetime.now().isoformat()
            }
        
        # Calculate signal weights
        signal_weights = {'Buy': 0, 'Sell': 0, 'Neutral': 0}
        for analysis in valid_analyses:
            signal = analysis.get('signal')
            confidence = analysis.get('confidence', '50%')
            try:
                conf_value = int(confidence.strip('%')) / 100
            except:
                conf_value = 0.5
            
            signal_weights[signal] += conf_value
        
        # Determine final signal
        if signal_weights['Buy'] > signal_weights['Sell'] and signal_weights['Buy'] > signal_weights['Neutral']:
            final_signal = "Buy"
            total_confidence = signal_weights['Buy']
        elif signal_weights['Sell'] > signal_weights['Buy'] and signal_weights['Sell'] > signal_weights['Neutral']:
            final_signal = "Sell"
            total_confidence = signal_weights['Sell']
        else:
            final_signal = "Neutral"
            total_confidence = signal_weights['Neutral']
        
        # Normalize confidence
        avg_confidence = int((total_confidence / len(valid_analyses)) * 100)
        
        return {
            "symbol": valid_analyses[0].get('symbol', 'Unknown'),
            "signal": final_signal,
            "confidence": f"{avg_confidence}%",
            "rationale": f"Overall {final_signal.lower()} sentiment based on {len(valid_analyses)} analyses",
            "sources_analyzed": len(valid_analyses),
            "analysis_details": valid_analyses,
            "timestamp": datetime.now().isoformat()
        }

    def combine_and_analyze_data(self, reddit_data: List[Dict[str, Any]], news_data: List[Dict[str, Any]], symbol: str) -> Dict[str, Any]:
        """
        Combine Reddit and News data, calculate metrics, and perform sentiment analysis
        """
        print(f"Combining and analyzing data for {symbol}")
        print(f"Reddit items: {len(reddit_data)}, News items: {len(news_data)}")
        
        # Calculate metrics for both sources
        reddit_with_metrics = self.calculate_metrics(reddit_data, 'reddit')
        news_with_metrics = self.calculate_metrics(news_data, 'news')
        
        # Combine all items
        all_items = reddit_with_metrics + news_with_metrics
        print(f"Total items with metrics: {len(all_items)}")
        
        # If no items, return neutral analysis
        if not all_items:
            return {
                "symbol": symbol,
                "signal": "Neutral",
                "confidence": "0%",
                "rationale": "No data available for analysis",
                "sources_analyzed": 0,
                "timestamp": datetime.now().isoformat()
            }
        
        # Analyze sentiment by source type
        analyses = []
        
        if reddit_with_metrics:
            print("Running AI analysis for Reddit posts...")
            reddit_analysis = self.analyze_sentiment_with_ai(reddit_with_metrics, symbol, 'reddit')
            analyses.append(reddit_analysis)
        
        if news_with_metrics:
            print("Running AI analysis for News items...")
            news_analysis = self.analyze_sentiment_with_ai(news_with_metrics, symbol, 'news')
            analyses.append(news_analysis)
        
        # Also analyze combined data
        print("Running AI analysis for combined data...")
        combined_analysis = self.analyze_sentiment_with_ai(all_items, symbol, 'combined')
        analyses.append(combined_analysis)
        
        # Aggregate analyses
        final_analysis = self.aggregate_analyses(analyses)
        
        # Add summary statistics
        final_analysis.update({
            "summary_stats": {
                "total_items": len(all_items),
                "reddit_items": len(reddit_with_metrics),
                "news_items": len(news_with_metrics),
                "avg_sentiment": round(np.mean([item['metrics']['sentiment_score'] for item in all_items]), 3),
                "avg_engagement": round(np.mean([item['metrics']['engagement_score'] for item in all_items]), 2)
            }
        })
        
        print(f"Final analysis completed for {symbol}")
        return final_analysis

# # Example usage:
# if __name__ == "__main__":
#     # Sample configuration
#     config = {
#         "reddit_credentials": {
#             "client_id": "your_client_id",
#             "client_secret": "your_client_secret",
#             "user_agent": "sentiment-analysis/1.0"
#         }
#     }
    
#     analyzer = TradingSentimentAnalyzer(config)
    
#     # Sample data (would come from your external fetchers)
#     sample_reddit_data = [
#         {
#             "title": "EURUSD showing strong bullish momentum",
#             "text": "The EURUSD pair has broken through resistance and looks bullish for the week ahead.",
#             "upvotes": 15,
#             "upvote_ratio": 0.85,
#             "subreddit_subscribers": 100000,
#             "created_at": datetime.now().isoformat()
#         }
#     ]
    
#     sample_news_data = [
#         {
#             "title": "ECB maintains rates, EUR strengthens",
#             "text": "The European Central Bank kept interest rates unchanged, leading to EUR strength across major pairs.",
#             "publisher": "forexlive.com",
#             "created_at": datetime.now().isoformat()
#         }
#     ]
    
#     # Combine and analyze
#     result = analyzer.combine_and_analyze_data(sample_reddit_data, sample_news_data, "EURUSD")
#     print("Analysis Result:")
#     print(json.dumps(result, indent=2))