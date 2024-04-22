# Crypto Real Time Trading System

## Overview
* Leveraged Transformer AI Model, Alpaca API for optimized trade execution and enhanced decision making
* Effectively extracted and processed live market data, enabling prompt decision-making and executing orders with high accuracy and efficiency

## Tools
Python, Tensorflow, Alpaca API, asyncio, 

## Structure
* **main.py:** Trigger streaming, plotting, loading data
* **market_minute.py:** Extract data in minute timeframe and save them under the price_bar_minute folder
* **streaming_trade.py:** Extract trade, quotes data in real-time manner and save them under the logging folder
* **live_plot.py:** Plot the trade data in live
* **transformer_model.py:** Layers of the BERT + Time embeddings model 
* **btc_order.py:** Submit order based on predicted price by transformer
* **model folder:** The final model is under this folder
* **requirements.txt:** Required packages is listed in this file
* **config.ini:** Configuration info is listed in this file
    
