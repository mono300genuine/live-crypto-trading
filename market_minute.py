import asyncio
import alpaca_trade_api as tradeapi
import configparser
import csv
import os
from alpaca.data.live import CryptoDataStream

class DataCollector:
    def __init__(self, config_path, symbols):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.api_key = self.config.get('alpaca', 'API_KEY')
        self.api_secret = self.config.get('alpaca', 'SECRET_KEY')
        self.symbols = symbols
        self.crypto_stream = None

    async def subscribe_data(self):
        tasks = []
        for symbol in self.symbols:
            tasks.append(self.crypto_stream.subscribe_bars(self.bar_callback, symbol))
        await asyncio.gather(*tasks)

    async def bar_callback(self, bar):
        row = [value for _, value in bar]
        symbol = row[0].lower().replace('/', '')
        filename = os.path.join("price_bar_minute", f"{symbol}_bar_data.csv")

        with open(filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)

            if csvfile.tell() == 0:
                header = [property_name for property_name, _ in bar]
                writer.writerow(header)

            row = [value for _, value in bar]
            writer.writerow(row)

    def run(self):
        log_folder = "price_bar_minute"
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)

        self.crypto_stream = CryptoDataStream(self.api_key, self.api_secret)
        for symbol in self.symbols:
            self.crypto_stream.subscribe_bars(self.bar_callback, symbol)
        self.crypto_stream.run()


    
