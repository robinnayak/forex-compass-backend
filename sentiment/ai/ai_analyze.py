import requests
import json
import re
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import time

logger = logging.getLogger(__name__)

class BaseAIAnalyzer(ABC):
    """Abstract base class for AI sentiment analyzers"""
    
    @abstractmethod
    def analyze_sentiment(self, items: List[Dict[str, Any]], symbol: str, source: str) -> Dict[str, Any]:
        """Analyze sentiment from given items"""
        pass
    
    @abstractmethod
    def create_prompt(self, items: List[Dict[str, Any]], symbol: str, source: str) -> str:
        """Create analysis prompt"""
        pass

class OllamaSentimentAnalyzer(BaseAIAnalyzer):
    """Ollama-based sentiment analyzer for financial content"""
    
    def __init__(self, 
                 model: str = "gemma3:latest",
                 base_url: str = "http://localhost:11434",
                 timeout: int = 300,
                 max_retries: int = 3):
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.api_url = f"{base_url}/api/generate"
        
    def analyze_sentiment(self, items: List[Dict[str, Any]], symbol: str, source: str) -> Dict[str, Any]:
        """
        Analyze sentiment using Ollama model running locally.
        """
        if not items:
            return self._create_error_response(symbol, "No items to analyze")
        
        prompt = self.create_prompt(items, symbol, source)
        
        for attempt in range(self.max_retries):
            try:
                response = self._call_ollama_api(prompt)
                analysis = self._parse_ai_response(response, symbol)
                return analysis
                
            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Ollama API request failed after {self.max_retries} attempts: {e}")
                    return self._create_error_response(symbol, f"Ollama API unavailable: {str(e)}")
                logger.warning(f"Attempt {attempt + 1} failed, retrying...")
                time.sleep(2 ** attempt)  # Exponential backoff
                
            except Exception as e:
                logger.error(f"Unexpected error in AI analysis: {e}")
                return self._create_error_response(symbol, f"Unexpected error: {str(e)}")
    
    def create_prompt(self, items: List[Dict[str, Any]], symbol: str, source: str) -> str:
        """Create a prompt for AI analysis"""
        examples = []
        for i, item in enumerate(items[:8]):  # Limit to 8 examples for context
            examples.append({
                "id": i + 1,
                "title": item.get('title', '')[:100],
                "preview": (item.get('text', '')[:200] + "...") if len(item.get('text', '')) > 200 else item.get('text', ''),
                "source": item.get('publisher') or item.get('subreddit', 'Unknown'),
                "date": item.get('created_at', '')[:10],
                "engagement": item.get('metrics', {}).get('engagement_score', 0),
                "sentiment": item.get('metrics', {}).get('sentiment_score', 0)
            })
        
        prompt = f"""
## ROLE: You are a senior financial analyst specializing in forex market sentiment analysis.

## TASK: Analyze the following {source} content about {symbol} and provide a comprehensive trading recommendation.

## CONTEXT:
- Symbol: {symbol}
- Source: {source}
- Items analyzed: {len(items)}
- Current date: {datetime.now().strftime('%Y-%m-%d')}

## CONTENT TO ANALYZE:
{json.dumps(examples, indent=2, ensure_ascii=False)}

## ANALYSIS GUIDELINES:
1. Assess overall sentiment (Bullish/Bearish/Neutral)
2. Evaluate sentiment strength (Weak/Moderate/Strong)
3. Consider engagement metrics and source credibility
4. Identify key themes or patterns
5. Provide probability estimates
6. Give concise but detailed rationale

## RESPONSE FORMAT:
Return ONLY valid JSON with this exact structure:
{{
  "symbol": "{symbol}",
  "signal": "Buy|Sell|Neutral",
  "confidence": "Low|Medium|High|Very High",
  "sentiment_score": 0.0 to 1.0,
  "buy_probability": "XX%",
  "sell_probability": "YY%",
  "neutral_probability": "ZZ%",
  "rationale": "2-3 sentence summary of key insights",
  "key_insights": ["insight1", "insight2", "insight3"],
  "risk_level": "Low|Medium|High",
  "timeframe": "Intraday|Short-term|Medium-term|Long-term",
  "sources_analyzed": {len(items)},
  "analysis_timestamp": "{datetime.now().isoformat()}"
}}

## IMPORTANT: 
- Return ONLY valid JSON, no additional text
- Use precise probability percentages that sum to 100%
- Base recommendations on actual content analysis
- Be objective and data-driven
"""
        return prompt
    
    def _call_ollama_api(self, prompt: str) -> str:
        """Make API call to Ollama"""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "top_p": 0.9,
                "max_tokens": 1000
            }
        }
        
        response = requests.post(self.api_url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        result = response.json()
        
        return result.get("response", "")
    
    def _parse_ai_response(self, ai_output: str, symbol: str) -> Dict[str, Any]:
        """Parse and validate AI response"""
        try:
            # Clean the response - remove markdown code blocks
            cleaned_output = re.sub(r'```json\s*|\s*```', '', ai_output).strip()
            
            # Try to parse as JSON
            analysis = json.loads(cleaned_output)
            
            # Validate required fields
            required_fields = ["symbol", "signal", "confidence", "rationale"]
            for field in required_fields:
                if field not in analysis:
                    raise ValueError(f"Missing required field: {field}")
            
            # Validate signal values
            if analysis["signal"] not in ["Buy", "Sell", "Neutral"]:
                raise ValueError(f"Invalid signal: {analysis['signal']}")
            
            # Validate confidence values
            valid_confidences = ["Low", "Medium", "High", "Very High"]
            if analysis["confidence"] not in valid_confidences:
                raise ValueError(f"Invalid confidence: {analysis['confidence']}")
            
            return analysis
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response as JSON: {e}")
            # Try to extract JSON from malformed response
            json_match = re.search(r'\{.*\}', ai_output, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except:
                    pass
            
            return self._create_error_response(symbol, "AI response could not be parsed as JSON", ai_output)
            
        except Exception as e:
            logger.error(f"Error validating AI response: {e}")
            return self._create_error_response(symbol, f"Error validating response: {str(e)}", ai_output)
    
    def _create_error_response(self, symbol: str, error_msg: str, raw_output: str = "") -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            "symbol": symbol,
            "signal": "Neutral",
            "confidence": "Low",
            "sentiment_score": 0.5,
            "buy_probability": "50%",
            "sell_probability": "50%",
            "neutral_probability": "0%",
            "rationale": error_msg,
            "key_insights": ["Analysis failed due to technical issues"],
            "risk_level": "High",
            "timeframe": "N/A",
            "sources_analyzed": 0,
            "analysis_timestamp": datetime.now().isoformat(),
            "error": True,
            "error_message": error_msg,
            "raw_output": raw_output[:500] if raw_output else ""
        }

class AnalysisAggregator:
    """Aggregates multiple AI analyses into a final recommendation"""
    
    def aggregate_analyses(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate multiple analyses into a final recommendation"""
        if not analyses:
            return self._create_empty_analysis()
        
        # Filter out error responses and invalid analyses
        valid_analyses = [
            a for a in analyses 
            if not a.get('error', False) and a.get('signal') in ['Buy', 'Sell', 'Neutral']
        ]
        
        if not valid_analyses:
            return self._create_error_analysis(analyses[0] if analyses else {})
        
        # Calculate weighted scores based on confidence
        confidence_weights = {"Low": 1, "Medium": 2, "High": 3, "Very High": 4}
        
        buy_score = sell_score = neutral_score = 0
        total_weight = 0
        
        for analysis in valid_analyses:
            weight = confidence_weights.get(analysis.get('confidence', 'Low'), 1)
            total_weight += weight
            
            if analysis['signal'] == 'Buy':
                buy_score += weight
            elif analysis['signal'] == 'Sell':
                sell_score += weight
            else:
                neutral_score += weight
        
        # Calculate probabilities
        buy_prob = int((buy_score / total_weight) * 100) if total_weight > 0 else 33
        sell_prob = int((sell_score / total_weight) * 100) if total_weight > 0 else 33
        neutral_prob = 100 - buy_prob - sell_prob
        
        # Determine final signal
        if buy_prob > sell_prob and buy_prob > neutral_prob:
            final_signal = "Buy"
        elif sell_prob > buy_prob and sell_prob > neutral_prob:
            final_signal = "Sell"
        else:
            final_signal = "Neutral"
        
        # Calculate average sentiment score
        sentiment_scores = [a.get('sentiment_score', 0.5) for a in valid_analyses]
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.5
        
        # Collect all insights and sources
        all_insights = []
        all_sources = set()
        
        for analysis in valid_analyses:
            all_insights.extend(analysis.get('key_insights', []))
            all_sources.add(analysis.get('source_type', 'unknown'))
        
        return {
            "symbol": valid_analyses[0].get('symbol', 'Unknown'),
            "signal": final_signal,
            "confidence": self._determine_aggregate_confidence(valid_analyses),
            "sentiment_score": round(avg_sentiment, 3),
            "buy_probability": f"{buy_prob}%",
            "sell_probability": f"{sell_prob}%",
            "neutral_probability": f"{neutral_prob}%",
            "rationale": f"Aggregated analysis from {len(valid_analyses)} sources showing {final_signal.lower()} sentiment",
            "key_insights": list(set(all_insights))[:5],  # Unique insights, max 5
            "risk_level": self._determine_aggregate_risk(valid_analyses),
            "timeframe": "Short-term",  # Default
            "sources_analyzed": len(valid_analyses),
            "source_types": list(all_sources),
            "analysis_timestamp": datetime.now().isoformat(),
            "component_analyses": [{
                "source_type": a.get('source_type', 'unknown'),
                "signal": a.get('signal'),
                "confidence": a.get('confidence'),
                "sentiment_score": a.get('sentiment_score', 0.5)
            } for a in valid_analyses]
        }
    
    def _determine_aggregate_confidence(self, analyses: List[Dict[str, Any]]) -> str:
        """Determine aggregate confidence level"""
        confidences = [a.get('confidence', 'Low') for a in analyses]
        confidence_values = {"Low": 1, "Medium": 2, "High": 3, "Very High": 4}
        
        avg_confidence = sum(confidence_values.get(c, 1) for c in confidences) / len(confidences)
        
        if avg_confidence >= 3.5:
            return "Very High"
        elif avg_confidence >= 2.5:
            return "High"
        elif avg_confidence >= 1.5:
            return "Medium"
        else:
            return "Low"
    
    def _determine_aggregate_risk(self, analyses: List[Dict[str, Any]]) -> str:
        """Determine aggregate risk level"""
        risk_levels = [a.get('risk_level', 'Medium') for a in analyses]
        risk_counts = {"Low": 0, "Medium": 0, "High": 0}
        
        for risk in risk_levels:
            risk_counts[risk] = risk_counts.get(risk, 0) + 1
        
        if risk_counts["High"] > risk_counts["Medium"] and risk_counts["High"] > risk_counts["Low"]:
            return "High"
        elif risk_counts["Low"] > risk_counts["Medium"] and risk_counts["Low"] > risk_counts["High"]:
            return "Low"
        else:
            return "Medium"
    
    def _create_empty_analysis(self) -> Dict[str, Any]:
        return {
            "symbol": "Unknown",
            "signal": "Neutral",
            "confidence": "Low",
            "sentiment_score": 0.5,
            "buy_probability": "50%",
            "sell_probability": "50%",
            "neutral_probability": "0%",
            "rationale": "No data available for analysis",
            "key_insights": ["Insufficient data from sources"],
            "risk_level": "High",
            "timeframe": "N/A",
            "sources_analyzed": 0,
            "analysis_timestamp": datetime.now().isoformat(),
            "error": True
        }
    
    def _create_error_analysis(self, original_analysis: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "symbol": original_analysis.get('symbol', 'Unknown'),
            "signal": "Neutral",
            "confidence": "Low",
            "sentiment_score": 0.5,
            "buy_probability": "50%",
            "sell_probability": "50%",
            "neutral_probability": "0%",
            "rationale": "No valid analyses available - all sources returned errors",
            "key_insights": ["All component analyses failed validation"],
            "risk_level": "High",
            "timeframe": "N/A",
            "sources_analyzed": 0,
            "analysis_timestamp": datetime.now().isoformat(),
            "error": True
        }

# Factory function for easy usage
def create_ai_analyzer(analyzer_type: str = "ollama", **kwargs) -> BaseAIAnalyzer:
    """Factory function to create AI analyzers"""
    if analyzer_type.lower() == "ollama":
        return OllamaSentimentAnalyzer(**kwargs)
    else:
        raise ValueError(f"Unsupported analyzer type: {analyzer_type}")

# # Example usage
# if __name__ == "__main__":
#     # Create analyzer
#     analyzer = create_ai_analyzer(
#         model="gemma3:latest",
#         base_url="http://localhost:11434",
#         timeout=300
#     )
    
#     # Create aggregator
#     aggregator = AnalysisAggregator()
    
#     # Example data
#     sample_items = [
#         {
#             "title": "EURUSD shows strong bullish momentum",
#             "text": "The EURUSD pair broke through key resistance levels...",
#             "publisher": "ForexLive",
#             "created_at": "2024-01-15T10:00:00",
#             "metrics": {"engagement_score": 15, "sentiment_score": 0.8}
#         }
#     ]
#     symbol = "EURUSD"
#     # Analyze sentiment
#     analysis = analyzer.analyze_sentiment(sample_items, symbol, "news")
#     print("Single analysis:", json.dumps(analysis, indent=2))
    
#     # Aggregate multiple analyses
#     analyses = [analysis]  # Normally you'd have multiple sources
#     final_analysis = aggregator.aggregate_analyses(analyses)
#     print("\nAggregated analysis:", json.dumps(final_analysis, indent=2))