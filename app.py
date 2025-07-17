from flask import Flask, render_template, request, jsonify
import requests
import os
import json
from datetime import datetime
from dotenv import load_dotenv

app = Flask(__name__)

# Enhanced conversation memory - stores full conversation history for each session
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


# FIXED: Enhanced conversation memory management
def get_conversation_history(session_id):
    """Get the conversation history for a session"""
    if session_id not in conversation_memory:
        conversation_memory[session_id] = {
            'messages': [],
            'last_topic': None,
            'last_intent': None
        }
    return conversation_memory[session_id]


def add_to_conversation(session_id, user_message, assistant_response, intent=None, topic=None):
    """Add a message pair to the conversation history"""
    history = get_conversation_history(session_id)

    # Add user message
    history['messages'].append({
        'role': 'user',
        'content': user_message,
        'timestamp': datetime.now().isoformat()
    })

    # Add assistant response
    history['messages'].append({
        'role': 'assistant',
        'content': assistant_response,
        'timestamp': datetime.now().isoformat()
    })

    # Update metadata
    if intent:
        history['last_intent'] = intent
    if topic:
        history['last_topic'] = topic

    # Keep only last 10 messages (5 pairs) to manage memory
    if len(history['messages']) > 10:
        history['messages'] = history['messages'][-10:]


def is_follow_up_query(user_input, session_id):
    """Enhanced follow-up detection"""
    follow_up_phrases = [
        'yes', 'tell me more', 'continue', 'more', 'go on', 'next', 'elaborate',
        'what else', 'anything else', 'more details', 'explain more', 'keep going',
        'and then', 'what about', 'how about', 'also', 'additionally'
    ]

    # Personal/contextual questions that need conversation history
    personal_questions = [
        'my name', 'what\'s my name', 'who am i', 'remember me', 'do you remember',
        'what did i say', 'what did i ask', 'earlier', 'before'
    ]

    user_lower = user_input.lower().strip()

    # Check if it's a personal/contextual question
    if any(phrase in user_lower for phrase in personal_questions):
        return True

    # Check if it's a short follow-up response
    if len(user_input.split()) <= 4 and any(phrase in user_lower for phrase in follow_up_phrases):
        return True

    # Check if there's recent conversation history
    history = get_conversation_history(session_id)
    if len(history['messages']) > 0:
        # If user input is very short and we have recent context, it's likely a follow-up
        if len(user_input.split()) <= 2 and len(history['messages']) >= 2:
            return True

    return False


# Function 3: FIXED Answer financial queries with proper conversation context
def answer_financial_query(user_input, intent, symbol=None, session_id="default"):
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

        # Get conversation history
        history = get_conversation_history(session_id)
        is_follow_up = is_follow_up_query(user_input, session_id)

        if intent == "unknown":
            # For non-financial topics, use conversation history but redirect to finance
            system_msg = """You are a helpful financial assistant. You remember personal details from previous conversations (names, preferences, etc.) and should use them when appropriate.

However, you should only discuss financial topics. For non-financial questions:
1. If it's a personal question (like "what's my name"), acknowledge the personal info from conversation history
2. Then politely redirect back to financial topics
3. Keep responses brief (max 30 words) and always end with a financial question or topic

Example: "Your name is [name from history]! Now, what financial goals can I help you achieve today?"

Be warm but focused on finance."""

            # Build messages with conversation history
            messages = [{"role": "system", "content": system_msg}]

            # Add recent conversation history for context (last 6 messages)
            if history['messages']:
                recent_messages = history['messages'][-6:]  # Last 3 pairs
                for msg in recent_messages:
                    messages.append({
                        "role": msg['role'],
                        "content": msg['content']
                    })

            # Add current user message
            messages.append({"role": "user", "content": user_input})
        else:
            # For financial topics, use full conversation context
            system_msg = """You are a knowledgeable financial assistant. You provide helpful, detailed information about finance, investing, cryptocurrencies, stocks, and economic concepts. 

IMPORTANT: You have access to conversation history and should remember personal details shared by users (like names, preferences, previous questions). Always use this information to provide personalized responses.

RESPONSE STYLE:
- Keep answers SHORT, CLEAR, and SIMPLE
- Maximum 3-4 sentences per response
- Use bullet points only when absolutely necessary
- Be direct and concise
- If the topic is complex, give a brief overview and ask if they want more details

When users ask about trading, investing, or financial strategies, provide:
- Key points only (not step-by-step unless requested)
- Essential information
- Brief practical advice
- Ask if they need more details

Be conversational and helpful. Use the conversation history to provide contextual responses."""

            # Build messages with conversation history
            messages = [{"role": "system", "content": system_msg}]

            # Add recent conversation history for context (last 6 messages to reduce context)
            if history['messages']:
                recent_messages = history['messages'][-6:]  # Last 3 pairs
                for msg in recent_messages:
                    messages.append({
                        "role": msg['role'],
                        "content": msg['content']
                    })

            # Add current user message
            messages.append({"role": "user", "content": user_input})

        payload = {
            "model": "llama3-8b-8192",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 150  # Reduced from 400 to force shorter responses
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
        print(f"DEBUG: Session ID: {session_id}")

        # Check if this is a follow-up query or personal question
        is_follow_up = is_follow_up_query(user_input, session_id)
        print(f"DEBUG: Is follow-up: {is_follow_up}")

        # Don't override intent - let the original classification handle it
        # Personal questions should be classified as "unknown" and handled with redirection

        # Route to appropriate function based on intent
        if intent == "crypto_price":
            if symbol:
                crypto_data = get_crypto_info(symbol)
                if crypto_data['success']:
                    response = f"""🪙 *{crypto_data['name']} ({crypto_data['symbol']})*

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

                    # Add to conversation history
                    add_to_conversation(session_id, user_input, response.strip(), intent,
                                        f"crypto data for {crypto_data['name']}")

                    return jsonify({'response': response.strip()})
                else:
                    error_response = f"❌ {crypto_data['error']}"
                    add_to_conversation(session_id, user_input, error_response, intent)
                    return jsonify({'response': error_response})
            else:
                error_response = '❌ Could not identify cryptocurrency symbol'
                add_to_conversation(session_id, user_input, error_response, intent)
                return jsonify({'response': error_response})

        elif intent == "stock_price":
            if symbol:
                stock_data = get_stock_info(symbol)
                if stock_data['success']:
                    change_emoji = "📈" if stock_data['change'] >= 0 else "📉"
                    response = f"""📊 *{stock_data['symbol']} Stock Information*

💰 *Current Price*: ${stock_data['current_price']:.2f}
{change_emoji} *Change*: {stock_data['change']:+.2f} ({stock_data['change_percent']}%)
🔓 *Open*: ${stock_data['open']:.2f}
⬆ *High*: ${stock_data['high']:.2f}
⬇ *Low*: ${stock_data['low']:.2f}
📊 *Volume*: {stock_data['volume']:,}

Last Updated: {stock_data['last_updated']}"""

                    # Add to conversation history
                    add_to_conversation(session_id, user_input, response.strip(), intent,
                                        f"stock data for {stock_data['symbol']}")

                    return jsonify({'response': response.strip()})
                else:
                    error_response = f"❌ {stock_data['error']}"
                    add_to_conversation(session_id, user_input, error_response, intent)
                    return jsonify({'response': error_response})
            else:
                error_response = '❌ Could not identify stock symbol'
                add_to_conversation(session_id, user_input, error_response, intent)
                return jsonify({'response': error_response})

        # Handle all other cases (define, unknown, and general financial queries)
        else:
            financial_response = answer_financial_query(user_input, intent, symbol, session_id)
            if financial_response['success']:
                response = financial_response['response']

                # Add to conversation history
                add_to_conversation(session_id, user_input, response, intent, user_input)

                return jsonify({'response': response})
            else:
                error_response = f"❌ {financial_response['error']}"
                add_to_conversation(session_id, user_input, error_response, intent)
                return jsonify({'response': error_response})

    except Exception as e:
        error_response = f'An error occurred: {str(e)}'
        return jsonify({'error': error_response})


# Optional: Add endpoint to clear conversation history
@app.route('/clear_history', methods=['POST'])
def clear_history():
    session_id = request.json.get('session_id', request.remote_addr or 'default')
    if session_id in conversation_memory:
        del conversation_memory[session_id]
    return jsonify({'success': True, 'message': 'Conversation history cleared'})


if __name__ == '__main__':
    app.run(debug=True)