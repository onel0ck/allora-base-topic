from flask import Flask, jsonify, request
import logging
from datetime import datetime
import requests
import random
from cachetools import TTLCache

logging.basicConfig(
   level=logging.INFO,
   format='%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s', 
   datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("main")

app = Flask(__name__)

CACHE_TIMINGS = {
   'ETH': 15,
   'BTC': 20, 
   'SOL': 35,
   'BNB': 40,
   'ARB': 45
}

PRICE_DECIMALS = {
   'ETH': 4,
   'BTC': 4,
   'SOL': 4,
   'BNB': 4,
   'ARB': 4
}

price_caches = {
   symbol: TTLCache(maxsize=1, ttl=timing)
   for symbol, timing in CACHE_TIMINGS.items()
}

TIMEFRAME_RANGES = {
   "10m": {
       "min_change": 0.08,
       "range": (-0.3, 0.3)
   },
   "20m": {
       "min_change": 0.08,
       "range": (-0.4, 0.4)
   },
   "1d": {
       "min_change": 0.6,
       "range": (-3.5, 3.5)
   }
}

TOPIC_MAP = {
   1: ("ETH", "10m"),
   2: ("ETH", "1d"),
   3: ("BTC", "10m"), 
   4: ("BTC", "1d"),
   5: ("SOL", "10m"),
   6: ("SOL", "1d"),
   7: ("ETH", "20m"),
   8: ("BNB", "20m"),
   9: ("ARB", "20m")
}

def get_current_price(symbol):
   cache = price_caches[symbol]
   cache_key = f"{symbol}_price"

   if cache_key in cache:
       cached_price = cache[cache_key]
       return cached_price

   try:
       url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}USDT"
       response = requests.get(url, timeout=5)

       if response.status_code == 200:
           current_price = float(response.json()["price"])
           cache[cache_key] = current_price
           logger.info(f"[API REQUEST] New price fetched for {symbol}: {current_price} (will be cached for {CACHE_TIMINGS[symbol]} seconds)")
           return current_price
       else:
           if cache_key in cache:
               old_price = cache[cache_key]
               logger.warning(f"Failed to get new price for {symbol}, using last known price: {old_price}")
               return old_price
           raise Exception(f"Failed to get price for {symbol}: {response.text}")

   except requests.exceptions.RequestException as e:
       if cache_key in cache:
           old_price = cache[cache_key]
           logger.warning(f"Error fetching price for {symbol}, using last known price: {old_price}. Error: {str(e)}")
           return old_price
       raise Exception(f"Failed to get price and no cached price available for {symbol}: {str(e)}")
   except Exception as e:
       logger.error(f"Error getting price for {symbol}: {str(e)}")
       raise

def generate_prediction(current_price, timeframe, token, worker_id):
   current_second = datetime.now().timestamp()
   random.seed(int((current_second * 1000 + worker_id) % 2**32))
   
   min_range, max_range = TIMEFRAME_RANGES[timeframe]["range"]
   
   worker_factor = hash(f"{worker_id}{current_second}") % 100 / 1000.0
   base_change = random.uniform(min_range, max_range)
   change_percent = base_change + worker_factor
   
   min_change = TIMEFRAME_RANGES[timeframe]["min_change"]
   if abs(change_percent) < min_change:
       direction = 1 if change_percent >= 0 else -1
       change_percent = direction * (min_change + random.uniform(0, 0.05))
   
   predicted_price = current_price * (1 + change_percent / 100)
   decimals = PRICE_DECIMALS.get(token, 2)
   predicted_price = round(predicted_price, decimals)
   
   final_price = predicted_price * (1 + random.uniform(-0.0001, 0.0001))
   final_price = round(final_price, decimals)
   
   logger.info(f"Worker {worker_id} | {timeframe} prediction | Base price: {current_price:.{decimals}f} | "
               f"Change: {change_percent:+.3f}% | Predicted: {final_price:.{decimals}f}")
   return final_price

@app.route("/inference/<int:topic_id>")
def get_inference(topic_id):
   if topic_id not in TOPIC_MAP:
       return jsonify({"error": "Unsupported topic ID"}), 400

   try:
       worker_id = int(request.args.get('worker_id', 0))
       token, timeframe = TOPIC_MAP[topic_id]

       current_price = get_current_price(token)
       prediction = generate_prediction(current_price, timeframe, token, worker_id)

       decimals = PRICE_DECIMALS.get(token, 2)
       return str(round(prediction, decimals))

   except Exception as e:
       logger.error(f"Error during inference: {str(e)}")
       return jsonify({"error": str(e)}), 500

@app.route("/health")
def health_check():
   status = {
       "status": "healthy",
       "cached_prices": {}
   }

   for symbol in CACHE_TIMINGS.keys():
       cache = price_caches[symbol]
       cache_key = f"{symbol}_price"
       if cache_key in cache:
           status["cached_prices"][symbol] = {
               "price": cache[cache_key],
           }

   return jsonify(status)

if __name__ == "__main__":
   app.run(host='0.0.0.0', port=8000)