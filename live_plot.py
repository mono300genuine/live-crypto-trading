from streaming_trade import AlpacaStream
import matplotlib.pyplot as plt
from alpaca_trade_api.stream import Stream
from datetime import datetime

plt.style.use('seaborn')

class AlpacaStream_Plot(AlpacaStream):
    def __init__(self, config_path, symbols):
        super().__init__(config_path, symbols)
        self.prices = {symbol: [] for symbol in symbols}
        self.timestamps = {symbol: [] for symbol in symbols}
        self.figures = {symbol: plt.figure() for symbol in symbols}
        self.axes = {symbol: fig.add_subplot() for symbol, fig in self.figures.items()}

    async def plot_trade(self, t):
        symbol = t['S']
        self.prices[symbol].append(t['p'])
        self.timestamps[symbol].append(datetime.fromtimestamp(t['t'].seconds))

        ax = self.axes[symbol]
        ax.clear()
        ax.plot(self.timestamps[symbol], self.prices[symbol])
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Price')
        ax.set_title(f'Live {symbol} Price Trend')
        plt.xticks(rotation=45)

        plt.pause(0.001)

    def start_stream(self):
        self.initialize_loggers()
        self.stream = Stream(self.api_key, self.api_secret, raw_data=True)
        self.subscribe_trades()

        for symbol in self.symbols:
            self.stream.subscribe_crypto_trades(self.plot_trade, symbol)

        for fig in self.figures.values():
            plt.figure(fig.number)
            plt.show(block=False)

        self.stream.run()

