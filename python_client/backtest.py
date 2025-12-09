class Backtest:
    def __init__(self, adapter, initial_cash=100000):
        self.adapter=adapter; self.cash=initial_cash; self.trades=[]
    def run(self):
        return {'cash': self.cash}
