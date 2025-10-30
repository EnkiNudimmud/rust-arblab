def test_backtest_stub():
    from python_client.backtest import Backtest
    bt=Backtest(None)
    assert 'cash' in bt.run()
