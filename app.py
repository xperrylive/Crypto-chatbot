from flask import Flask, render_template, request, jsonify
import requests
import os
import json
from datetime import datetime
from dotenv import load_dotenv
app = Flask(__name__)

# Simple conversation memory - stores last context for each session
conversation_memory = {}
load_dotenv()

# Your existing function - unchanged
def analyze_user_input(user_input):
    llm_api_key = os.getenv('llm_api_key')
    if not llm_api_key:
        raise ValueError("Missing GROQ_API_KEY in environment")

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {llm_api_key}",
        "Content-Type": "application/json"
    }

    system_msg = "You are a financial assistant. You classify finance-related user input and extract trading symbols."

    user_prompt = f"""
Your task is to classify a user's financial question and extract:

- intent: one of ["crypto_price", "stock_price", "define", "unknown"]
- symbol: the trading symbol or keyword (e.g. BTC, ETH, AAPL, inflation)
- raw_entity: the original asset or term (e.g. bitcoin, Apple)

Instructions:
- If the user asks for ANY crypto-related info (price, market cap, ATH, ATL, volume, etc.), use intent: "crypto_price" and convert the name to its *standard crypto symbol* (e.g. "bitcoin" → "BTC").
- If the user asks for ANY stock-related info (price, market cap, volume, etc.), use intent: "stock_price" and convert the company name to its *stock ticker* (e.g. "Apple" → "AAPL").
- If the user asks for a definition or explanation OR asks about general financial topics like trading, investing, strategies, etc., use intent: "define" and keep the keyword as-is.
- If it's not finance-related, return intent: "unknown" and both symbol/raw_entity: null

Examples:
- "what's the price of bitcoin?" → crypto_price, BTC, bitcoin
- "bitcoin ATH" → crypto_price, BTC, bitcoin
- "ethereum market cap" → crypto_price, ETH, ethereum
- "SOL volume" → crypto_price, SOL, solana
- "apple stock price?" → stock_price, AAPL, Apple
- "Tesla market cap" → stock_price, TSLA, Tesla
- "AAPL volume" → stock_price, AAPL, Apple
- "define inflation" → define, inflation, inflation
- "how can i start trading" → define, trading, trading
- "what are investment strategies" → define, investment, investment
- "yes" → define, general, general
- "tell me more" → define, general, general
- "how tall is Mount Everest?" → unknown, null, null

Respond ONLY with JSON like:
{{"intent": "crypto_price", "symbol": "BTC", "raw_entity": "bitcoin"}}

Now classify:
"{user_input}"
"""

    payload = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        message = response.json()["choices"][0]["message"]["content"]

        # Parse the strict JSON response
        return json.loads(message)

    except Exception as e:
        print("Error from Groq:", e)
        return {
            "intent": "unknown",
            "symbol": None,
            "raw_entity": None
        }


# Function 1: Get crypto price and info using CoinGecko
def get_crypto_info(symbol):
    try:
        # Try the symbol directly first
        url = f"https://api.coingecko.com/api/v3/coins/{symbol.lower()}"
        response = requests.get(url, timeout=10)

        if response.status_code != 200:
            # If direct lookup fails, search for the coin
            search_url = f"https://api.coingecko.com/api/v3/search?query={symbol}"
            search_response = requests.get(search_url, timeout=10)

            if search_response.status_code == 200:
                search_data = search_response.json()
                if search_data['coins']:
                    coin_id = search_data['coins'][0]['id']
                    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
                    response = requests.get(url, timeout=10)
                else:
                    return {'success': False, 'error': 'Cryptocurrency not found'}
            else:
                return {'success': False, 'error': 'Cryptocurrency not found'}

        if response.status_code == 200:
            data = response.json()

            current_price = data['market_data']['current_price']['usd']
            price_change_24h = data['market_data']['price_change_percentage_24h']
            market_cap = data['market_data']['market_cap']['usd']
            volume_24h = data['market_data']['total_volume']['usd']

            # Get ATH and ATL data
            ath = data['market_data'].get('ath', {}).get('usd')
            ath_date = data['market_data'].get('ath_date', {}).get('usd')
            atl = data['market_data'].get('atl', {}).get('usd')
            atl_date = data['market_data'].get('atl_date', {}).get('usd')

            return {
                'success': True,
                'name': data['name'],
                'symbol': data['symbol'].upper(),
                'current_price': current_price,
                'price_change_24h': round(price_change_24h, 2) if price_change_24h else 0,
                'market_cap': market_cap,
                'volume_24h': volume_24h,
                'ath': ath,
                'ath_date': ath_date,
                'atl': atl,
                'atl_date': atl_date,
                'last_updated': data['last_updated']
            }
        else:
            return {'success': False, 'error': 'Cryptocurrency not found'}

    except Exception as e:
        return {'success': False, 'error': f'Error fetching crypto data: {str(e)}'}


# Function 2: Get stock price and info using Alpha Vantage
def get_stock_info(symbol):
    try:
        # You'll need to get your own Alpha Vantage API key from https://www.alphavantage.co/support/#api-key
        alpha_vantage_key = os.getenv("alpha_vantage_key")  # Replace with your actual API key

        if not alpha_vantage_key:
            return {'success': False, 'error': 'Please set up your Alpha Vantage API key'}

        # Get quote data
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={alpha_vantage_key}"
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            data = response.json()

            # Check for API errors
            if 'Error Message' in data:
                return {'success': False, 'error': 'Stock symbol not found'}

            if 'Note' in data:
                return {'success': False, 'error': 'API rate limit exceeded. Please try again later.'}

            if 'Global Quote' in data and data['Global Quote']:
                quote = data['Global Quote']

                # Check if the quote data is empty
                if not quote or '05. price' not in quote:
                    return {'success': False, 'error': 'Stock symbol not found'}

                current_price = float(quote['05. price'])
                change = float(quote['09. change'])
                change_percent = quote['10. change percent'].replace('%', '')

                return {
                    'success': True,
                    'symbol': quote['01. symbol'],
                    'current_price': current_price,
                    'change': change,
                    'change_percent': change_percent,
                    'previous_close': float(quote['08. previous close']),
                    'open': float(quote['02. open']),
                    'high': float(quote['03. high']),
                    'low': float(quote['04. low']),
                    'volume': int(quote['06. volume']),
                    'last_updated': quote['07. latest trading day']
                }
            else:
                return {'success': False, 'error': 'Stock symbol not found'}
        else:
            return {'success': False, 'error': 'Failed to fetch stock data'}

    except Exception as e:
        return {'success': False, 'error': f'Error fetching stock data: {str(e)}'}


# Function 3: Answer financial queries and greetings using Groq - FIXED
def answer_financial_query(user_input, intent, symbol=None, session_id="default", previous_context=None):
    try:
        # Handle common greetings with short responses
        greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening', 'howdy']
        if any(greeting in user_input.lower() for greeting in greetings):
            return {'success': True,
                    'response': '👋 Hello! Ask me about crypto prices, stock prices, or financial topics!'}

        llm_api_key = os.getenv('llm_api_key')

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {llm_api_key}",
            "Content-Type": "application/json"
        }

        # Initialize messages array for all cases
        messages = []

        if intent == "unknown":
            system_msg = "You are a helpful financial assistant. Give a very short, polite response (max 15 words) redirecting to financial topics."
            user_prompt = f"The user asked: '{user_input}'. Give a brief response redirecting to financial topics."

            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_prompt}
            ]
        else:
            system_msg = """You are a knowledgeable financial assistant. You provide helpful, detailed information about finance, investing, cryptocurrencies, stocks, and economic concepts. 

When users ask about trading, investing, or financial strategies, provide comprehensive guidance including:
- Step-by-step instructions
- Risk management tips
- Practical advice
- Specific recommendations

Be conversational and helpful. If a user responds with "yes", "tell me more", "continue", or asks for more information, continue the conversation naturally and provide the next logical step or more detailed information based on the previous context."""

            # Build conversation context
            messages = [{"role": "system", "content": system_msg}]

            # Add previous context if available
            if previous_context:
                messages.append({"role": "assistant", "content": previous_context})

            messages.append({"role": "user", "content": user_input})

        payload = {
            "model": "llama3-8b-8192",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 300
        }

        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()

        message = response.json()["choices"][0]["message"]["content"]
        return {'success': True, 'response': message}

    except Exception as e:
        return {'success': False, 'error': f'Error generating response: {str(e)}'}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json.get('message', '').strip()
        # Use IP address as session ID if no session_id provided
        session_id = request.json.get('session_id', request.remote_addr or 'default')

        if not user_input:
            return jsonify({'error': 'Please enter a message'})

        # Analyze user input using your existing function
        analysis = analyze_user_input(user_input)
        intent = analysis.get('intent')
        symbol = analysis.get('symbol')
        raw_entity = analysis.get('raw_entity')

        print(f"DEBUG: User input: '{user_input}'")
        print(f"DEBUG: Analysis result: {analysis}")

        # Get previous context from memory
        previous_context = conversation_memory.get(session_id, {}).get('last_response')

        # Check if this is a follow-up response (yes, tell me more, continue, etc.)
        follow_up_words = ['yes', 'tell me more', 'continue', 'more', 'go on', 'next', 'elaborate']
        is_follow_up = any(word in user_input.lower() for word in follow_up_words) and len(user_input.split()) <= 3

        # Route to appropriate function based on intent
        if intent == "crypto_price":
            if symbol:
                crypto_data = get_crypto_info(symbol)
                if crypto_data['success']:
                    response = f"""
🪙 *{crypto_data['name']} ({crypto_data['symbol']})*

💰 *Current Price*: ${crypto_data['current_price']:,.2f}
📈 *24h Change*: {crypto_data['price_change_24h']}%
🏦 *Market Cap*: ${crypto_data['market_cap']:,.0f}
📊 *24h Volume*: ${crypto_data['volume_24h']:,.0f}"""

                    # Add ATH and ATL if available
                    if crypto_data.get('ath'):
                        ath_date = crypto_data['ath_date'][:10] if crypto_data.get('ath_date') else 'N/A'
                        response += f"\n🚀 *All-Time High*: ${crypto_data['ath']:,.2f} ({ath_date})"

                    if crypto_data.get('atl'):
                        atl_date = crypto_data['atl_date'][:10] if crypto_data.get('atl_date') else 'N/A'
                        response += f"\n📉 *All-Time Low*: ${crypto_data['atl']:,.6f} ({atl_date})"

                    response += f"\n\n*Last Updated: {crypto_data['last_updated']}*"

                    # Store in conversation memory
                    conversation_memory[session_id] = {
                        'last_response': response.strip(),
                        'last_topic': f"crypto data for {crypto_data['name']}"
                    }

                    return jsonify({'response': response.strip()})
                else:
                    return jsonify({'response': f"❌ {crypto_data['error']}"})
            else:
                return jsonify({'response': '❌ Could not identify cryptocurrency symbol'})

        elif intent == "stock_price":
            if symbol:
                stock_data = get_stock_info(symbol)
                if stock_data['success']:
                    change_emoji = "📈" if stock_data['change'] >= 0 else "📉"
                    response = f"""
📊 *{stock_data['symbol']} Stock Information*

💰 *Current Price*: ${stock_data['current_price']:.2f}
{change_emoji} *Change*: {stock_data['change']:+.2f} ({stock_data['change_percent']}%)
🔓 *Open*: ${stock_data['open']:.2f}
⬆ *High*: ${stock_data['high']:.2f}
⬇ *Low*: ${stock_data['low']:.2f}
📊 *Volume*: {stock_data['volume']:,}

Last Updated: {stock_data['last_updated']}
                    """

                    # Store in conversation memory
                    conversation_memory[session_id] = {
                        'last_response': response.strip(),
                        'last_topic': f"stock data for {stock_data['symbol']}"
                    }

                    return jsonify({'response': response.strip()})
                else:
                    return jsonify({'response': f"❌ {stock_data['error']}"})
            else:
                return jsonify({'response': '❌ Could not identify stock symbol'})

        # Handle all other cases (define, unknown, and general financial queries)
        else:
            # For follow-up responses, use previous context
            context_to_use = previous_context if is_follow_up else None

            financial_response = answer_financial_query(user_input, intent, symbol, session_id, context_to_use)
            if financial_response['success']:
                response = financial_response['response']

                # Store in conversation memory
                conversation_memory[session_id] = {
                    'last_response': response,
                    'last_topic': user_input if not is_follow_up else conversation_memory.get(session_id, {}).get(
                        'last_topic', user_input)
                }

                return jsonify({'response': response})
            else:
                return jsonify({'response': f"❌ {financial_response['error']}"})

    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'})


if __name__ == '__main__':
    app.run(debug=True)