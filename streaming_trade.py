import os
import configparser
import logging
import asyncio
from alpaca_trade_api.stream import Stream

class AlpacaStream:
    def __init__(self, config_path, symbols):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.api_key = self.config.get('alpaca', 'API_KEY')
        self.api_secret = self.config.get('alpaca', 'SECRET_KEY')
        self.symbols = symbols
        self.stream = None

    async def log_quote(self, q):
        symbol = q['S']
        symbol = symbol.lower().replace('/', '')
        logger = logging.getLogger(symbol)
        logger.info('quote %s', q)

    async def log_trade(self, t):
        symbol = t['S']
        symbol = symbol.lower().replace('/', '')
        logger = logging.getLogger(symbol)
        logger.info('trade %s', t)

    def start_stream(self):
        self.initialize_loggers()
        self.stream = Stream(self.api_key, self.api_secret, raw_data=True)
        self.subscribe_quotes()
        self.subscribe_trades()

        self.stream.run()

    def stop_stream(self):
        if self.stream:
            self.stream.close()

    def initialize_loggers(self):
        log_folder = "logging"
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)

        for symbol in self.symbols:
            symbol = symbol.lower().replace('/', '')
            logger = logging.getLogger(symbol)
            logger.setLevel(logging.INFO)
            handler = logging.FileHandler(os.path.join(log_folder, f"{symbol}.log"))
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

    def subscribe_quotes(self):
        for symbol in self.symbols:
            self.stream.subscribe_crypto_quotes(self.log_quote, symbol)

    def subscribe_trades(self):
        for symbol in self.symbols:
            self.stream.subscribe_crypto_trades(self.log_trade, symbol)





