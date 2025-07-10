from flask import Flask, render_template, request, jsonify
import requests
import json
import re
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# API Configuration
GEMINI_API_KEY= "AIzaSyDtBavYsdZgy4CNr4jqQzNqRhPApQIZGT4"
ALPHA_VANTAGE_API_KEY= "E7GA52EAIO85QVS3"
GEMINI_API_URL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        # Check if this is a financial query using keyword detection
        is_financial_query = is_financial_request(user_message)
        
        if is_financial_query:
            # Handle financial queries
            return handle_financial_query(user_message)
        else:
            # Handle general questions with Gemini
            return handle_general_query(user_message)
    
    except Exception as e:
        print(f"Error in /ask route: {str(e)}")
        return jsonify({'response': 'Sorry, I encountered an error. Please try again.'})

def is_financial_request(message):
    """Check if the message is specifically asking for financial data/prices"""
    message_lower = message.lower()
    
    # First, check if it's clearly NOT a price request
    general_question_indicators = [
        'what is', 'what are', 'explain', 'how does', 'how do', 'tell me about',
        'definition', 'meaning', 'history', 'learn about', 'understand',
        'difference between', 'compare', 'vs', 'versus', 'which is better',
        'pros and cons', 'advantages', 'disadvantages', 'risks', 'benefits',
        'how to invest', 'how to trade', 'how to buy', 'should i invest',
        'good investment', 'safe investment', 'risky investment', 'advice',
        'recommend', 'suggest', 'opinion', 'thoughts', 'news', 'latest',
        'update', 'analysis', 'review', 'performance', 'forecast', 'prediction'
    ]
    
    # If it contains general question indicators, it's likely not a price request
    for indicator in general_question_indicators:
        if indicator in message_lower:
            return False
    
    # Now check for specific price request patterns
    price_request_patterns = [
        # Direct price questions
        r'\b(what.*price|current.*price|price.*of|how much.*cost|cost.*of)\b',
        r'\b(what.*worth|worth.*of|value.*of|current.*value)\b',
        r'\b(price.*today|today.*price|current.*trading|trading.*at)\b',
        
        # Specific format requests
        r'\b(show.*price|get.*price|find.*price|check.*price)\b',
        r'\b\$\d+|\d+\s*dollar|\d+\s*usd\b',  # Contains actual price format
        
        # Stock/crypto with price context
        r'\b(stock|crypto|bitcoin|ethereum|btc|eth)\s+(price|cost|value|worth)\b',
        r'\b(price|cost|value|worth)\s+(stock|crypto|bitcoin|ethereum|btc|eth)\b',
        
        # Company name + price context
        r'\b(apple|tesla|google|microsoft|amazon|meta|netflix|nvidia)\s+(stock\s+)?(price|cost|value|worth)\b',
        r'\b(price|cost|value|worth)\s+(apple|tesla|google|microsoft|amazon|meta|netflix|nvidia)\b'
    ]
    
    # Check for price request patterns
    for pattern in price_request_patterns:
        if re.search(pattern, message_lower):
            return True
    
    # Check for stock symbols in isolation (likely price requests)
    stock_symbols = ['aapl', 'tsla', 'googl', 'msft', 'amzn', 'meta', 'nflx', 'nvda', 'amd']
    crypto_symbols = ['btc', 'eth', 'ada', 'sol', 'dot', 'xrp', 'ltc', 'bch', 'bnb', 'matic']
    
    # Only treat as price request if symbol is mentioned without explanatory context
    words = message_lower.split()
    for symbol in stock_symbols + crypto_symbols:
        if symbol in words:
            # Check if it's in a question context (likely general question)
            question_words = ['what', 'how', 'why', 'when', 'where', 'explain', 'tell']
            if any(q_word in words for q_word in question_words):
                return False
            # If just the symbol or with price-related words, it's likely a price request
            return True
    

def handle_financial_query(user_message):
    """Handle financial queries with improved parsing"""
    try:
        # Try to extract financial info using simple pattern matching first
        symbol, asset_type = extract_financial_info_simple(user_message)
        
        if not symbol:
            # Fallback to Gemini for complex queries
            gemini_response = analyze_intent_with_gemini(user_message)
            if gemini_response:
                symbol = extract_symbol(gemini_response)
                asset_type = extract_asset_type(gemini_response)
        
        if not symbol:
            return jsonify({'response': 'I couldn\'t identify the specific stock or cryptocurrency. Please try asking with a clearer symbol like "Apple stock price" or "Bitcoin price".'})
        
        # Get price data based on asset type
        if asset_type == 'crypto':
            price_data = get_crypto_price(symbol)
        else:
            price_data = get_stock_price(symbol)
        
        if price_data:
            return jsonify({'response': price_data})
        else:
            return jsonify({'response': f'Sorry, I couldn\'t find price information for {symbol}. Please check the symbol and try again.'})
    
    except Exception as e:
        print(f"Error in handle_financial_query: {str(e)}")
        return jsonify({'response': 'Sorry, I encountered an error while fetching financial data. Please try again.'})

def extract_financial_info_simple(message):
    """Extract financial information using simple pattern matching"""
    message_lower = message.lower()
    
    # Common company name to symbol mapping
    company_symbols = {
        'apple': ('AAPL', 'stock'),
        'tesla': ('TSLA', 'stock'),
        'google': ('GOOGL', 'stock'),
        'microsoft': ('MSFT', 'stock'),
        'amazon': ('AMZN', 'stock'),
        'meta': ('META', 'stock'),
        'facebook': ('META', 'stock'),
        'netflix': ('NFLX', 'stock'),
        'nvidia': ('NVDA', 'stock'),
        'amd': ('AMD', 'stock'),
        'bitcoin': ('bitcoin', 'crypto'),
        'ethereum': ('ethereum', 'crypto'),
        'btc': ('bitcoin', 'crypto'),
        'eth': ('ethereum', 'crypto'),
        'cardano': ('cardano', 'crypto'),
        'ada': ('cardano', 'crypto'),
        'solana': ('solana', 'crypto'),
        'sol': ('solana', 'crypto'),
        'polkadot': ('polkadot', 'crypto'),
        'dot': ('polkadot', 'crypto'),
        'ripple': ('ripple', 'crypto'),
        'xrp': ('ripple', 'crypto'),
        'litecoin': ('litecoin', 'crypto'),
        'ltc': ('litecoin', 'crypto'),
        'binance': ('binancecoin', 'crypto'),
        'bnb': ('binancecoin', 'crypto'),
        'polygon': ('polygon', 'crypto'),
        'matic': ('polygon', 'crypto')
    }
    
    # Check for exact matches
    for name, (symbol, asset_type) in company_symbols.items():
        if name in message_lower:
            return symbol, asset_type
    
    # Check for stock symbols in uppercase
    stock_pattern = r'\b([A-Z]{1,5})\b'
    matches = re.findall(stock_pattern, message)
    if matches:
        return matches[0], 'stock'
    
    return None, None

def handle_general_query(user_message):
    """Handle general questions using Gemini with enhanced capabilities"""
    try:
        response = ask_gemini_general(user_message)
        if response:
            return jsonify({'response': response})
        else:
            return jsonify({'response': 'I\'m sorry, I couldn\'t process your question at the moment. Please try again.'})
    
    except Exception as e:
        print(f"Error in handle_general_query: {str(e)}")
        return jsonify({'response': 'Sorry, I encountered an error. Please try again.'})

def ask_gemini_general(user_message):
    """Ask Gemini for general questions with enhanced prompting"""
    try:
        # Determine the type of question and adjust the prompt accordingly
        question_type = categorize_question(user_message)
        
        # Check if it's a financial topic but not a price request
        is_financial_topic = is_financial_topic_general(user_message)
        
        if is_financial_topic:
            prompt = f"""
            You are a knowledgeable financial education assistant. Provide a clear, informative response about: {user_message}
            
            Guidelines:
            - Give direct, educational information without repeating the question
            - Keep it concise but comprehensive (under 200 words) (you determine how long the answer should be)
            - If relevant, mention that real-time price data is also available
            - Use simple language and explain technical terms
            """
        elif question_type == 'factual':
            prompt = f"""
            Provide a clear, factual answer about: {user_message}
            
            Guidelines:
            - Give direct information without repeating the question
            - Include key facts and relevant details
            - Keep it concise (under 200 words)
            - Use authoritative tone
            """
        elif question_type == 'creative':
            prompt = f"""
            Create an engaging, creative response for: {user_message}
            
            Guidelines:
            - Be imaginative and helpful
            - Don't repeat the request
            - Focus on delivering creative content
            """
        elif question_type == 'advice':
            prompt = f"""
            Provide thoughtful, practical advice for: {user_message}
            
            Guidelines:
            - Give actionable advice without repeating the question
            - Consider multiple perspectives
            - Be supportive and helpful
            - Keep it practical and realistic
            """
        elif question_type == 'technical':
            prompt = f"""
            Explain clearly: {user_message}
            
            Guidelines:
            - Provide technical explanation without repeating the question
            - Use appropriate terminology but explain complex concepts
            - Include examples if helpful
            - Make it accessible to the user
            """
        else:
            prompt = f"""
            Provide a helpful response about: {user_message}
            
            Guidelines:
            - Answer directly without repeating the question
            - Be conversational and friendly
            - Give useful information
            - Keep it under 300 words unless more detail is needed
            """
        
        headers = {
            'Content-Type': 'application/json',
        }
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 1024,
            }
        }
        
        response = requests.post(
            f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
            headers=headers,
            json=payload,
            timeout=15
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'candidates' in result and len(result['candidates']) > 0:
                return result['candidates'][0]['content']['parts'][0]['text']
        else:
            print(f"Gemini API error: {response.status_code} - {response.text}")
        
        return None
        
    except Exception as e:
        print(f"Error calling Gemini API for general query: {str(e)}")
        return None

def is_financial_topic_general(message):
    """Check if the message is about financial topics but not requesting prices"""
    financial_topic_keywords = [
        'cryptocurrency', 'blockchain', 'bitcoin', 'ethereum', 'stock market',
        'investing', 'investment', 'trading', 'portfolio', 'diversification',
        'bull market', 'bear market', 'recession', 'inflation', 'economics',
        'finance', 'financial', 'nasdaq', 'nyse', 'dow jones', 'sp500',
        'mutual funds', 'etf', 'bonds', 'dividend', 'market cap', 'volatility'
    ]
    
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in financial_topic_keywords)

def categorize_question(message):
    """Categorize the type of question to adjust response style"""
    message_lower = message.lower()
    
    # Creative questions
    creative_keywords = ['write', 'create', 'story', 'poem', 'joke', 'creative', 'imagine', 'design']
    if any(keyword in message_lower for keyword in creative_keywords):
        return 'creative'
    
    # Advice questions
    advice_keywords = ['should i', 'what should', 'how to', 'advice', 'recommend', 'suggest', 'help me']
    if any(keyword in message_lower for keyword in advice_keywords):
        return 'advice'
    
    # Technical questions
    technical_keywords = ['how does', 'explain', 'algorithm', 'code', 'programming', 'technical', 'science', 'math']
    if any(keyword in message_lower for keyword in technical_keywords):
        return 'technical'
    
    # Factual questions
    factual_keywords = ['what is', 'who is', 'when did', 'where is', 'definition', 'fact', 'history']
    if any(keyword in message_lower for keyword in factual_keywords):
        return 'factual'
    
    return 'general'

def analyze_intent_with_gemini(user_message):
    """Use Gemini to analyze user intent and extract relevant information"""
    try:
        prompt = f"""
        Analyze this message and extract financial information: "{user_message}"
        
        Return ONLY a JSON response with these fields:
        - "symbol": the stock ticker or cryptocurrency symbol (e.g., "AAPL", "bitcoin", "BTC")
        - "type": either "stock" or "crypto"
        - "intent": "price_inquiry" if asking about price, otherwise "other"
        
        Examples:
        - "What's the price of Apple stock?" → {{"symbol": "AAPL", "type": "stock", "intent": "price_inquiry"}}
        - "How much is Bitcoin?" → {{"symbol": "bitcoin", "type": "crypto", "intent": "price_inquiry"}}
        - "Tesla stock price" → {{"symbol": "TSLA", "type": "stock", "intent": "price_inquiry"}}
        
        Return only the JSON, no other text.
        """
        
        headers = {
            'Content-Type': 'application/json',
        }
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }]
        }
        
        response = requests.post(
            f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
            headers=headers,
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'candidates' in result and len(result['candidates']) > 0:
                text = result['candidates'][0]['content']['parts'][0]['text']
                return text.strip()
        
        return None
        
    except Exception as e:
        print(f"Error calling Gemini API: {str(e)}")
        return None

def extract_symbol(gemini_response):
    """Extract symbol from Gemini response"""
    try:
        # Try to parse as JSON first
        data = json.loads(gemini_response)
        return data.get('symbol', '').upper()
    except:
        # Fallback: extract using regex
        symbol_match = re.search(r'"symbol":\s*"([^"]+)"', gemini_response)
        if symbol_match:
            return symbol_match.group(1).upper()
        return None

def extract_asset_type(gemini_response):
    """Extract asset type from Gemini response"""
    try:
        # Try to parse as JSON first
        data = json.loads(gemini_response)
        return data.get('type', 'stock').lower()
    except:
        # Fallback: extract using regex
        type_match = re.search(r'"type":\s*"([^"]+)"', gemini_response)
        if type_match:
            return type_match.group(1).lower()
        return 'stock'

def get_crypto_price(symbol):
    """Get cryptocurrency price from CoinGecko API"""
    try:
        # Convert common symbols to CoinGecko IDs
        crypto_map = {
            'BTC': 'bitcoin',
            'ETH': 'ethereum',
            'ADA': 'cardano',
            'DOT': 'polkadot',
            'XRP': 'ripple',
            'LTC': 'litecoin',
            'BCH': 'bitcoin-cash',
            'BNB': 'binancecoin',
            'SOL': 'solana',
            'MATIC': 'polygon'
        }
        
        crypto_id = crypto_map.get(symbol.upper(), symbol.lower())
        
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={crypto_id}&vs_currencies=usd&include_24hr_change=true"
        
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if crypto_id in data:
                price = data[crypto_id]['usd']
                change_24h = data[crypto_id].get('usd_24h_change', 0)
                
                change_text = f"📈 +{change_24h:.2f}%" if change_24h > 0 else f"📉 {change_24h:.2f}%"
                
                return f"💰 **{symbol.upper()}** is currently **${price:,.2f}** USD\n{change_text} (24h change)\n\n🕐 *Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
        
        return None
        
    except Exception as e:
        print(f"Error getting crypto price: {str(e)}")
        return None

def get_stock_price(symbol):
    """Get stock price from Alpha Vantage API"""
    try:
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
        
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'Global Quote' in data and data['Global Quote']:
                quote = data['Global Quote']
                price = float(quote['05. price'])
                change = float(quote['09. change'])
                change_percent = quote['10. change percent'].replace('%', '')
                
                change_text = f"📈 +${abs(change):.2f} (+{change_percent}%)" if change > 0 else f"📉 -${abs(change):.2f} ({change_percent}%)"
                
                return f"📊 **{symbol.upper()}** is currently **${price:.2f}** USD\n{change_text} (daily change)\n\n🕐 *Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
        
        return None
        
    except Exception as e:
        print(f"Error getting stock price: {str(e)}")
        return None

def handle_financial_query(user_message):
    """Handle financial queries with improved parsing"""
    try:
        # Try to extract financial info using simple pattern matching first
        symbol, asset_type = extract_financial_info_simple(user_message)
        
        if not symbol:
            # Fallback to Gemini for complex queries
            gemini_response = analyze_intent_with_gemini(user_message)
            if gemini_response:
                symbol = extract_symbol(gemini_response)
                asset_type = extract_asset_type(gemini_response)
        
        if not symbol:
            return jsonify({'response': 'I couldn\'t identify the specific stock or cryptocurrency. Please try asking with a clearer symbol like "Apple stock price" or "Bitcoin price".'})
        
        # Get price data based on asset type
        if asset_type == 'crypto':
            price_data = get_crypto_price(symbol)
        else:
            price_data = get_stock_price(symbol)
        
        if price_data:
            return jsonify({'response': price_data})
        else:
            return jsonify({'response': f'Sorry, I couldn\'t find price information for {symbol}. Please check the symbol and try again.'})
    
    except Exception as e:
        print(f"Error in handle_financial_query: {str(e)}")
        return jsonify({'response': 'Sorry, I encountered an error while fetching financial data. Please try again.'})

def extract_financial_info_simple(message):
    """Extract financial information using simple pattern matching"""
    message_lower = message.lower()
    
    # Common company name to symbol mapping
    company_symbols = {
        'apple': ('AAPL', 'stock'),
        'tesla': ('TSLA', 'stock'),
        'google': ('GOOGL', 'stock'),
        'microsoft': ('MSFT', 'stock'),
        'amazon': ('AMZN', 'stock'),
        'meta': ('META', 'stock'),
        'facebook': ('META', 'stock'),
        'netflix': ('NFLX', 'stock'),
        'nvidia': ('NVDA', 'stock'),
        'amd': ('AMD', 'stock'),
        'bitcoin': ('bitcoin', 'crypto'),
        'ethereum': ('ethereum', 'crypto'),
        'btc': ('bitcoin', 'crypto'),
        'eth': ('ethereum', 'crypto'),
        'cardano': ('cardano', 'crypto'),
        'ada': ('cardano', 'crypto'),
        'solana': ('solana', 'crypto'),
        'sol': ('solana', 'crypto'),
        'polkadot': ('polkadot', 'crypto'),
        'dot': ('polkadot', 'crypto'),
        'ripple': ('ripple', 'crypto'),
        'xrp': ('ripple', 'crypto'),
        'litecoin': ('litecoin', 'crypto'),
        'ltc': ('litecoin', 'crypto'),
        'binance': ('binancecoin', 'crypto'),
        'bnb': ('binancecoin', 'crypto'),
        'polygon': ('polygon', 'crypto'),
        'matic': ('polygon', 'crypto')
    }
    
    # Check for exact matches
    for name, (symbol, asset_type) in company_symbols.items():
        if name in message_lower:
            return symbol, asset_type
    
    # Check for stock symbols in uppercase
    stock_pattern = r'\b([A-Z]{1,5})\b'
    matches = re.findall(stock_pattern, message)
    if matches:
        return matches[0], 'stock'
    
    return None, None

def handle_general_query(user_message):
    """Handle general questions using Gemini with enhanced capabilities"""
    try:
        response = ask_gemini_general(user_message)
        if response:
            return jsonify({'response': response})
        else:
            return jsonify({'response': 'I\'m sorry, I couldn\'t process your question at the moment. Please try again.'})
    
    except Exception as e:
        print(f"Error in handle_general_query: {str(e)}")
        return jsonify({'response': 'Sorry, I encountered an error. Please try again.'})

def ask_gemini_general(user_message):
    """Ask Gemini for general questions with enhanced prompting"""
    try:
        # Determine the type of question and adjust the prompt accordingly
        question_type = categorize_question(user_message)
        
        if question_type == 'factual':
            prompt = f"""
            You are a knowledgeable AI assistant. Please provide a factual, accurate answer to the following question. 
            Include relevant details but keep the response concise (under 300 words).
            
            Question: {user_message}
            """
        elif question_type == 'creative':
            prompt = f"""
            You are a creative AI assistant. Please provide an engaging, creative response to the following request.
            Feel free to be imaginative while being helpful.
            
            Request: {user_message}
            """
        elif question_type == 'advice':
            prompt = f"""
            You are a helpful AI assistant providing advice. Please give thoughtful, practical advice for the following question.
            Consider multiple perspectives and be supportive in your response.
            
            Question: {user_message}
            """
        elif question_type == 'technical':
            prompt = f"""
            You are a technical AI assistant. Please provide a clear, detailed explanation for the following technical question.
            Use appropriate technical terminology but explain complex concepts clearly.
            
            Question: {user_message}
            """
        else:
            prompt = f"""
            You are a helpful AI assistant. Please answer the following question in a friendly, conversational way. 
            Provide accurate information and be as helpful as possible. Keep your response under 300 words unless more detail is needed.
            
            If you think the user might be asking about financial data (stocks, crypto prices), let them know you can help with real-time financial information.
            
            Question: {user_message}
            """
        
        headers = {
            'Content-Type': 'application/json',
        }
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 1024,
            }
        }
        
        response = requests.post(
            f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
            headers=headers,
            json=payload,
            timeout=15
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'candidates' in result and len(result['candidates']) > 0:
                return result['candidates'][0]['content']['parts'][0]['text']
        else:
            print(f"Gemini API error: {response.status_code} - {response.text}")
        
        return None
        
    except Exception as e:
        print(f"Error calling Gemini API for general query: {str(e)}")
        return None

def categorize_question(message):
    """Categorize the type of question to adjust response style"""
    message_lower = message.lower()
    
    # Creative questions
    creative_keywords = ['write', 'create', 'story', 'poem', 'joke', 'creative', 'imagine', 'design']
    if any(keyword in message_lower for keyword in creative_keywords):
        return 'creative'
    
    # Advice questions
    advice_keywords = ['should i', 'what should', 'how to', 'advice', 'recommend', 'suggest', 'help me']
    if any(keyword in message_lower for keyword in advice_keywords):
        return 'advice'
    
    # Technical questions
    technical_keywords = ['how does', 'explain', 'algorithm', 'code', 'programming', 'technical', 'science', 'math']
    if any(keyword in message_lower for keyword in technical_keywords):
        return 'technical'
    
    # Factual questions
    factual_keywords = ['what is', 'who is', 'when did', 'where is', 'definition', 'fact', 'history']
    if any(keyword in message_lower for keyword in factual_keywords):
        return 'factual'
    
    return 'general'

def analyze_intent_with_gemini(user_message):
    """Use Gemini to analyze user intent and extract relevant information"""
    try:
        prompt = f"""
        Analyze this message and extract financial information: "{user_message}"
        
        Return ONLY a JSON response with these fields:
        - "symbol": the stock ticker or cryptocurrency symbol (e.g., "AAPL", "bitcoin", "BTC")
        - "type": either "stock" or "crypto"
        - "intent": "price_inquiry" if asking about price, otherwise "other"
        
        Examples:
        - "What's the price of Apple stock?" → {{"symbol": "AAPL", "type": "stock", "intent": "price_inquiry"}}
        - "How much is Bitcoin?" → {{"symbol": "bitcoin", "type": "crypto", "intent": "price_inquiry"}}
        - "Tesla stock price" → {{"symbol": "TSLA", "type": "stock", "intent": "price_inquiry"}}
        
        Return only the JSON, no other text.
        """
        
        headers = {
            'Content-Type': 'application/json',
        }
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }]
        }
        
        response = requests.post(
            f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
            headers=headers,
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'candidates' in result and len(result['candidates']) > 0:
                text = result['candidates'][0]['content']['parts'][0]['text']
                return text.strip()
        
        return None
        
    except Exception as e:
        print(f"Error calling Gemini API: {str(e)}")
        return None

def extract_symbol(gemini_response):
    """Extract symbol from Gemini response"""
    try:
        # Try to parse as JSON first
        data = json.loads(gemini_response)
        return data.get('symbol', '').upper()
    except:
        # Fallback: extract using regex
        symbol_match = re.search(r'"symbol":\s*"([^"]+)"', gemini_response)
        if symbol_match:
            return symbol_match.group(1).upper()
        return None

def extract_asset_type(gemini_response):
    """Extract asset type from Gemini response"""
    try:
        # Try to parse as JSON first
        data = json.loads(gemini_response)
        return data.get('type', 'stock').lower()
    except:
        # Fallback: extract using regex
        type_match = re.search(r'"type":\s*"([^"]+)"', gemini_response)
        if type_match:
            return type_match.group(1).lower()
        return 'stock'

def get_crypto_price(symbol):
    """Get cryptocurrency price from CoinGecko API"""
    try:
        # Convert common symbols to CoinGecko IDs
        crypto_map = {
            'BTC': 'bitcoin',
            'ETH': 'ethereum',
            'ADA': 'cardano',
            'DOT': 'polkadot',
            'XRP': 'ripple',
            'LTC': 'litecoin',
            'BCH': 'bitcoin-cash',
            'BNB': 'binancecoin',
            'SOL': 'solana',
            'MATIC': 'polygon'
        }
        
        crypto_id = crypto_map.get(symbol.upper(), symbol.lower())
        
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={crypto_id}&vs_currencies=usd&include_24hr_change=true"
        
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if crypto_id in data:
                price = data[crypto_id]['usd']
                change_24h = data[crypto_id].get('usd_24h_change', 0)
                
                change_text = f"📈 +{change_24h:.2f}%" if change_24h > 0 else f"📉 {change_24h:.2f}%"
                
                return f"💰 **{symbol.upper()}** is currently **${price:,.2f}** USD\n{change_text} (24h change)\n\n🕐 *Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
        
        return None
        
    except Exception as e:
        print(f"Error getting crypto price: {str(e)}")
        return None

def get_stock_price(symbol):
    """Get stock price from Alpha Vantage API"""
    try:
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
        
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'Global Quote' in data and data['Global Quote']:
                quote = data['Global Quote']
                price = float(quote['05. price'])
                change = float(quote['09. change'])
                change_percent = quote['10. change percent'].replace('%', '')
                
                change_text = f"📈 +${abs(change):.2f} (+{change_percent}%)" if change > 0 else f"📉 -${abs(change):.2f} ({change_percent}%)"
                
                return f"📊 **{symbol.upper()}** is currently **${price:.2f}** USD\n{change_text} (daily change)\n\n🕐 *Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
        
        return None
        
    except Exception as e:
        print(f"Error getting stock price: {str(e)}")
        return None

if __name__ == '__main__':
    app.run(debug=True)