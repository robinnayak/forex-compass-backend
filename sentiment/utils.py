from typing import Dict, Any, Optional  
import logging 
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

valid_pairs = {"EURUSD", "USDJPY", "GBPUSD", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"}

def load_symbol_metadata(standardized_symbol: str) -> Dict[str, Any]:
    print(f"Loading metadata for symbol: {standardized_symbol}")
    try:
        metadata = {
            "EURUSD": {  # Changed from "EUR/USD" to "EURUSD"
                "keywords": ["EUR/USD", "Euro Dollar", "ECB", "European Central Bank"],
                "subs": ["Forex", "investing", "economics"],
                "category": "forex",
                "regions": ["EU", "US"],
                "note": "Major currency pair"
            },
            "USDJPY": {  # Changed from "USD/JPY" to "USDJPY"
                "keywords": ["USD/JPY", "US Dollar", "Japanese Yen", "BOJ", "Bank of Japan"],
                "subs": ["Forex", "investing", "economics"],
                "category": "forex",
                "regions": ["US", "JP"],
                "note": "Major currency pair"
            },
            "GBPUSD": {
                "keywords": ["GBP/USD", "Cable", "British Pound", "Bank of England", "BOE"],
                "subs": ["Forex", "investing", "economics"],
                "category": "forex",
                "regions": ["UK", "US"],
                "note": "Major currency pair"
            },
            "USDCHF": {
                "keywords": ["USD/CHF", "Swiss Franc", "SNB", "Swiss National Bank"],
                "subs": ["Forex", "investing", "economics"],
                "category": "forex",
                "regions": ["US", "CH"],
                "note": "Major currency pair"
            },
            # Add more currency pairs as needed
        }
        return metadata.get(standardized_symbol, {})
    except Exception as e:
        logger.error(f"Error loading symbol metadata: {e}") 
        return {}
    

def standardize_currency_pair(pair: str) -> Optional[str]:
    """Standardize currency pair input, handling variations like EUR/USD, EURUSD, eur usd, etc."""
    if not pair or not isinstance(pair, str):
        return None
    
    # Convert to uppercase and remove extra spaces
    clean_pair = pair.upper().strip()
    
    # Handle text that contains the pair (e.g., "I am eurusd")
    # Extract currency pair using regex pattern for 6-letter currency codes
    pair_match = re.search(r'([A-Z]{3}[A-Z]{3})', clean_pair.replace(" ", "").replace("/", ""))
    if pair_match:
        clean_pair = pair_match.group(1)
    
    # Remove any non-alphabetic characters (spaces, slashes, punctuation)
    clean_pair = re.sub(r'[^A-Z]', '', clean_pair)
    
    # Check if it's exactly 6 characters (standard currency pair format)
    if len(clean_pair) != 6:
        logger.warning(f"Invalid currency pair format: {pair}")
        return None
    
    # Check if it's a valid pair
    if clean_pair in valid_pairs:
        return clean_pair  # Return as "EURUSD" format
    
    # Also check if the reverse pair is valid (some might input USDEUR instead of EURUSD)
    reverse_pair = clean_pair[3:] + clean_pair[:3]
    if reverse_pair in valid_pairs:
        return reverse_pair
    
    logger.warning(f"Invalid currency pair: {pair}")
    return None


# Test function to verify the fix
def test_functions():
    test_cases = ["EURUSD", "EUR/USD", "eur usd", "I am eurusd", "USDJPY", "USD/JPY"]
    
    print("Testing standardization and metadata loading:")
    print("=" * 60)
    
    for test_case in test_cases:
        standardized = standardize_currency_pair(test_case)
        if standardized:
            metadata = load_symbol_metadata(standardized)
            print(f"Input: '{test_case}' -> Standardized: '{standardized}' -> Metadata: {bool(metadata)}")
            if metadata:
                print(f"  Keywords: {metadata.get('keywords', [])[:3]}")
        else:
            print(f"Input: '{test_case}' -> Invalid pair")
    
    print("=" * 60)


# Run test if executed directly
if __name__ == "__main__":
    test_functions()