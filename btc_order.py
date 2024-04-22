from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from datetime import datetime
import numpy as np
import pandas as pd
import logging
import asyncio
import configparser
from transformer_model import load_model


# ENABLE LOGGING - options, DEBUG,INFO, WARNING?

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Alpaca Trading Client
config = configparser.ConfigParser()
config.read('config.ini')

api_key = config.get('alpaca', 'api_key')
api_secret = config.get('alpaca', 'api_secret')

trading_client = TradingClient(api_key, api_secret, paper=True)

# Trading variables
qty_to_trade = 1

async def main(trading_pair):
    '''
    Function to get the latest asset data and check the position trade condition
    '''
    # closes all positions AND also cancels all open orders
    # trading_client.close_all_positions(cancel_orders=True)
    # logger.info("Closed all positions")
    while True:
        logger.info('--------------------------------------------')
        # load model
        model = load_model()
        # get predicted price
        global predicted_price
        logger.info("Getting bar data for {0} starting from {1}".format(trading_pair, "2020-01-01 00:00:00"))
        predicted_price = predict_func(model, trading_pair)
        logger.info("Predicted Price is {0}".format(predicted_price))

        l1 = loop.create_task(check_condition(trading_pair))
        await asyncio.wait([l1])
        # operate trading every 1 hour
        await asyncio.sleep(3600)


def predict_func(model, trading_pair):
    # Defining Bar data request parameters
    request_params = CryptoBarsRequest(
        symbol_or_symbols=[trading_pair],
        timeframe=TimeFrame.Hour,
        start="2020-01-01 00:00:00")

    # Get the bar data from Alpaca
    data_client = CryptoHistoricalDataClient()
    bars_df = data_client.get_crypto_bars(request_params).df

    # change to returns to feed into the model
    df = pd.DataFrame()
    df['open'] = bars_df['open'].pct_change()
    df['high'] = bars_df['high'].pct_change()
    df['low'] = bars_df['low'].pct_change()
    df['close'] = bars_df['close'].pct_change()
    df['volume'] = bars_df['volume'].pct_change()

    # Normalization
    min_return = min(df[['open', 'high', 'low', 'close']].min(axis=0))
    max_return = max(df[['open', 'high', 'low', 'close']].max(axis=0))

    df['open'] = (df['open'] - min_return) / (max_return - min_return)
    df['high'] = (df['high'] - min_return) / (max_return - min_return)
    df['low'] = (df['low'] - min_return) / (max_return - min_return)
    df['close'] = (df['close'] - min_return) / (max_return - min_return)

    min_volume = df['volume'].min(axis=0)
    max_volume = df['volume'].max(axis=0)
    df.loc[:, 'volume'] = (df.loc[:, 'volume'] - min_volume) / (max_volume - min_volume)

    global current_price
    current_price = bars_df.iloc[-1]['close']

    # df to array
    current_input = df.values

    # get sequential input with length 128
    seq_input = [current_input[-128:]]
    seq_input = np.array(seq_input)

    # get predicted price
    global predicted_price
    predicted_returns = model.predict(seq_input)
    predicted_returns = predicted_returns*(max_return - min_return) + min_return
    predicted_price = current_price * (1 + predicted_returns)
    return predicted_price


async def check_condition(trading_pair):

    global current_position, current_price, predicted_price
    current_position = get_positions(trading_pair)
    logger.info(f"Current Price is: {current_price}")
    logger.info(f"Current Position is: {current_position}")

    # If we do not have a position and the current price is less than the predicted price, place a market buy order
    if float(current_position) <= 0.01 and current_price < predicted_price:
        logger.info("Placing Buy Order")
        buy_order = await post_alpaca_order('buy', trading_pair)
        if buy_order:
            logger.info("Buy Order Placed")

    # If we do have a position and the current price is greater than the predicted price, place a market sell order
    if float(current_position) >= 0.01 and current_price > predicted_price:
        logger.info("Placing Sell Order")
        sell_order = await post_alpaca_order('sell', trading_pair)
        if sell_order:
            logger.info("Sell Order Placed")


async def post_alpaca_order(side, trading_pair):
    '''
    Post an order to Alpaca
    '''
    try:
        if side == 'buy':
            market_order_data = MarketOrderRequest(
                symbol=trading_pair,
                qty=qty_to_trade,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.GTC
            )
            buy_order = trading_client.submit_order(
                order_data=market_order_data
            )
            return buy_order
        else:
            market_order_data = MarketOrderRequest(
                symbol=trading_pair,
                qty=current_position,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC
            )
            sell_order = trading_client.submit_order(
                order_data=market_order_data
            )
            return sell_order
    except Exception as e:
        logger.exception(
            "There was an issue posting the order to Alpaca: {0}".format(e))
        return False


def get_positions(trading_pair):
    positions = trading_client.get_all_positions()
    global current_position
    for p in positions:
        if p.symbol == trading_pair:
            current_position = p.qty
            return current_position
    # no position
    return 0


loop = asyncio.get_event_loop()
trading_pair = 'BTC/USD'  # Set the trading pair here
loop.run_until_complete(main(trading_pair))
loop.close()


