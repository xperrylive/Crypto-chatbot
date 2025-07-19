from flask import Flask, render_template, request, jsonify
import requests
import os
import json
from datetime import datetime
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import pandas as pd
import io
import base64

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

- intent: one of ["crypto_price", "stock_price", "chart", "define", "unknown"]
- symbol: the trading symbol or keyword (e.g. BTC, ETH, AAPL, inflation)
- raw_entity: the original asset or term (e.g. bitcoin, Apple)
- time_period: for charts only, one of ["1d", "7d", "30d", "90d", "1y"] or null
- asset_type: for charts only, either "crypto" or "stock"

Instructions:
- If user asks for ANY crypto-related info (price, market cap, ATH, etc.), use intent: "crypto_price"
- If user asks for ANY stock-related info (price, market cap, volume, etc.), use intent: "stock_price"  
- If user asks for CHARTS/GRAPHS/VISUALIZATION of crypto/stock prices, use intent: "chart" and extract:
  * time_period: ONLY one of ["1d", "7d", "30d", "90d", "1y"] (default to "30d" if not specified)
  * asset_type: "crypto" for cryptocurrencies (bitcoin, ethereum, etc.) or "stock" for companies (Apple, Tesla, etc.)
- If user asks for definitions or general financial topics, use intent: "define"
- If not finance-related, return intent: "unknown"

Chart Examples:
- "show me bitcoin chart" → chart, BTC, bitcoin, "30d", "crypto"
- "AAPL 1 year chart" → chart, AAPL, Apple, "1y", "stock"
- "ethereum price chart last week" → chart, ETH, ethereum, "7d", "crypto"
- "chart tesla stock 3 months" → chart, TSLA, Tesla, "90d", "stock"

Respond ONLY with JSON like:
{{"intent": "chart", "symbol": "BTC", "raw_entity": "bitcoin", "time_period": "30d", "asset_type": "crypto"}}

Now classify: "{user_input}"
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
        return json.loads(message)
    except Exception as e:
        print("Error from Groq:", e)
        return {"intent": "unknown", "symbol": None, "raw_entity": None, "time_period": None, "asset_type": None}


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


# FIXED: Answer financial queries with proper token limits for complete responses
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

            # Use lower token limit for redirects
            max_tokens = 100
        else:
            # For financial topics, use full conversation context
            system_msg = """You are a knowledgeable financial assistant. You provide helpful, concise information about finance, investing, cryptocurrencies, stocks, and economic concepts. 

IMPORTANT: You have access to conversation history and should remember personal details shared by users (like names, preferences, previous questions). Always use this information to provide personalized responses.

RESPONSE REQUIREMENTS:
- Keep responses SHORT and COMPLETE (2-3 sentences maximum)
- Be direct and to the point
- Cover only the most essential points
- End with a complete thought, not mid-sentence
- No lengthy explanations unless specifically requested

For investment tips, strategies, or comparisons:
- Give key points only, be concise
- Focus on the most important advice
- Keep comparisons brief but complete

Be conversational and helpful. Use conversation history for context."""

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

            # Use lower token limit for short, complete answers
            max_tokens = 120

        payload = {
            "model": "llama3-8b-8192",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": max_tokens,
            "stop": None  # Remove any stop sequences that might cut off responses
        }

        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()

        message = response.json()["choices"][0]["message"]["content"].strip()

        # Check if response seems incomplete (ends abruptly without proper punctuation)
        if len(message) > 50 and not message.endswith(('.', '!', '?', '"', "'")):
            # Try to get a more complete response with slightly higher token limit
            payload["max_tokens"] = 150
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            message = response.json()["choices"][0]["message"]["content"].strip()

        return {'success': True, 'response': message}

    except Exception as e:
        return {'success': False, 'error': f'Error generating response: {str(e)}'}


def create_price_chart(symbol, time_period, asset_type):
    try:
        # Calculate date range
        days_map = {"1d": 1, "7d": 7, "30d": 30, "90d": 90, "1y": 365}
        days = days_map.get(time_period, 30)

        if asset_type == "crypto":
            # Get crypto historical data from CoinGecko
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            # Try direct symbol first, then search
            url = f"https://api.coingecko.com/api/v3/coins/{symbol.lower()}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': days,
                'interval': 'daily' if days > 1 else 'hourly'
            }

            response = requests.get(url, params=params, timeout=10)

            if response.status_code != 200:
                # Search for coin ID
                search_url = f"https://api.coingecko.com/api/v3/search?query={symbol}"
                search_response = requests.get(search_url, timeout=10)
                if search_response.status_code == 200:
                    search_data = search_response.json()
                    if search_data['coins']:
                        coin_id = search_data['coins'][0]['id']
                        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
                        response = requests.get(url, params=params, timeout=10)
                    else:
                        return {'success': False, 'error': 'Cryptocurrency not found'}
                else:
                    return {'success': False, 'error': 'Cryptocurrency not found'}

            if response.status_code == 200:
                data = response.json()
                prices = data['prices']

                # Convert to DataFrame
                df = pd.DataFrame(prices, columns=['timestamp', 'price'])
                df['date'] = pd.to_datetime(df['timestamp'], unit='ms')

                # Create chart
                plt.figure(figsize=(12, 6))
                plt.plot(df['date'], df['price'], linewidth=2, color='#f7931a')
                plt.title(f'{symbol.upper()} Price Chart ({time_period})', fontsize=16, fontweight='bold')
                plt.xlabel('Date')
                plt.ylabel('Price (USD)')
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45)

                # Format y-axis
                if df['price'].max() > 1000:
                    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
                else:
                    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.2f}'))

                plt.tight_layout()

                # Save to base64
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
                img_buffer.seek(0)
                chart_b64 = base64.b64encode(img_buffer.read()).decode()
                plt.close()

                return {
                    'success': True,
                    'chart_data': chart_b64,
                    'current_price': df['price'].iloc[-1],
                    'price_change': ((df['price'].iloc[-1] - df['price'].iloc[0]) / df['price'].iloc[0] * 100),
                    'symbol': symbol.upper()
                }

        elif asset_type == "stock":
            # Get stock historical data from Alpha Vantage
            alpha_vantage_key = os.getenv("alpha_vantage_key")
            if not alpha_vantage_key:
                return {'success': False, 'error': 'Alpha Vantage API key required for stock charts'}

            # Use TIME_SERIES_DAILY for historical data
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': symbol,
                'apikey': alpha_vantage_key,
                'outputsize': 'full'
            }

            response = requests.get(url, params=params, timeout=15)

            if response.status_code == 200:
                data = response.json()

                if 'Error Message' in data:
                    return {'success': False, 'error': 'Stock symbol not found'}

                if 'Note' in data:
                    return {'success': False, 'error': 'API rate limit exceeded'}

                time_series = data.get('Time Series (Daily)', {})
                if not time_series:
                    return {'success': False, 'error': 'No stock data available'}

                # Convert to DataFrame and filter by date range
                df_data = []
                cutoff_date = datetime.now() - timedelta(days=days)

                for date_str, values in time_series.items():
                    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                    if date_obj >= cutoff_date:
                        df_data.append({
                            'date': date_obj,
                            'price': float(values['4. close'])
                        })

                df = pd.DataFrame(df_data).sort_values('date')

                if df.empty:
                    return {'success': False, 'error': 'No recent stock data available'}

                # Create chart
                plt.figure(figsize=(12, 6))
                plt.plot(df['date'], df['price'], linewidth=2, color='#1f77b4')
                plt.title(f'{symbol.upper()} Stock Price Chart ({time_period})', fontsize=16, fontweight='bold')
                plt.xlabel('Date')
                plt.ylabel('Price (USD)')
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.2f}'))
                plt.tight_layout()

                # Save to base64
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
                img_buffer.seek(0)
                chart_b64 = base64.b64encode(img_buffer.read()).decode()
                plt.close()

                return {
                    'success': True,
                    'chart_data': chart_b64,
                    'current_price': df['price'].iloc[-1],
                    'price_change': ((df['price'].iloc[-1] - df['price'].iloc[0]) / df['price'].iloc[0] * 100),
                    'symbol': symbol.upper()
                }

        return {'success': False, 'error': 'Invalid asset type'}

    except Exception as e:
        return {'success': False, 'error': f'Error creating chart: {str(e)}'}



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
        time_period = analysis.get('time_period')
        asset_type = analysis.get('asset_type')

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

        elif intent == "chart":
            if symbol:
                time_period = analysis.get('time_period', '30d')
                asset_type = analysis.get('asset_type')

                # Validate time period
                valid_periods = ["1d", "7d", "30d", "90d", "1y"]
                if time_period not in valid_periods:
                    error_response = f"❌ Invalid time period. I can only generate charts for: {', '.join(valid_periods)}"
                    add_to_conversation(session_id, user_input, error_response, intent)
                    return jsonify({'response': error_response})

                if not asset_type:
                    error_response = "❌ Could not determine if this is a crypto or stock asset"
                    add_to_conversation(session_id, user_input, error_response, intent)
                    return jsonify({'response': error_response})

                chart_result = create_price_chart(symbol, time_period, asset_type)

                if chart_result['success']:
                    change_emoji = "📈" if chart_result['price_change'] >= 0 else "📉"
                    response = f"""📊 *{chart_result['symbol']} Price Chart ({time_period})*

💰 *Current Price*: ${chart_result['current_price']:.2f}
{change_emoji} *Period Change*: {chart_result['price_change']:+.2f}%

📈 *Chart generated successfully!*"""

                    # Add to conversation history
                    add_to_conversation(session_id, user_input, response.strip(), intent,
                                        f"chart for {chart_result['symbol']}")

                    return jsonify({
                        'response': response.strip(),
                        'chart': chart_result['chart_data']
                    })
                else:
                    error_response = f"❌ {chart_result['error']}"
                    add_to_conversation(session_id, user_input, error_response, intent)
                    return jsonify({'response': error_response})
            else:
                error_response = '❌ Could not identify asset symbol for chart'
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