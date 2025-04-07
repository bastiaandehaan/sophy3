import vectorbt as vbt
import pandas as pd
import MetaTrader5 as mt5

vbt.settings.array_wrapper['freq'] = '1m'


def get_historical_data(symbol, timeframe=mt5.TIMEFRAME_M1, start_date=None,
                        end_date=None):
    if not mt5.initialize():
        print("MT5 initialization failed")
        print(mt5.last_error())
        return None
    else:
        print("MT5 successfully initialized")
    if start_date is None:
        start_date = pd.Timestamp.now() - pd.Timedelta(days=365)
    if end_date is None:
        end_date = pd.Timestamp.now()
    start_date = start_date.to_pydatetime()
    end_date = end_date.to_pydatetime()
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    if rates is None or len(rates) == 0:
        print(f"Failed to get data for {symbol}")
        print(mt5.last_error())
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df.rename(columns={'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close',
                       'volume': 'volume'}, inplace=True)
    return df


def trader_tom_strategy(data, symbol, initial_capital=80000):
    data.index = data.index.tz_localize('UTC').tz_convert('Europe/Brussels')
    if symbol == 'GER40.cash':
        pre_start, pre_end = '08:00', '08:59'
        open_start, open_end = '09:00', '09:01'
        point_value = 1
        tp_points = 6
    elif symbol == 'US30.cash':
        pre_start, pre_end = '14:30', '15:29'
        open_start, open_end = '15:30', '15:31'
        point_value = 1
        tp_points = 9

    pre_market = data.between_time(pre_start, pre_end)
    open_data = data.between_time(open_start, open_end)
    print(
        f"{symbol} - Pre-market time range: {pre_market.index.min()} to {pre_market.index.max()}")
    print(
        f"{symbol} - Opening time range: {open_data.index.min()} to {open_data.index.max()}")

    daily_pre_market = pre_market.groupby(pre_market.index.date)
    highs = daily_pre_market['high'].max()
    lows = daily_pre_market['low'].min()
    print(f"{symbol} - Pre-market days: {len(daily_pre_market.groups)}")

    open_df = open_data.groupby(open_data.index.date).first()
    open_df['pre_high'] = open_df.index.map(lambda x: highs.get(x, None))
    open_df['pre_low'] = open_df.index.map(lambda x: lows.get(x, None))
    print(f"{symbol} - Opening days: {len(open_df)}")

    entries_long = open_df['close'] > open_df['pre_high']
    entries_short = open_df['close'] < open_df['pre_low']
    entries = entries_long.astype(int)
    entries.loc[entries_short & ~entries_long] = -1
    print(f"{symbol} - Sample trades (first 5 days):")
    print(open_df[['open', 'close', 'pre_high', 'pre_low']].head())
    print(entries.head())

    sl_points = 9
    sl_stop = sl_points / open_df['close']
    tp_stop = tp_points / open_df['close']

    size = initial_capital / (open_df['close'] * point_value)

    pf = vbt.Portfolio.from_signals(close=open_df['close'], entries=(entries != 0),
        direction='both', sl_stop=sl_stop, tp_stop=tp_stop, size=size,
        price=open_df['open'], fees=0.0001)

    total_return_percent = pf.total_return()
    total_profit = total_return_percent * initial_capital
    if symbol == 'US30.cash':
        total_profit *= 0.9

    return pf, total_profit


if __name__ == "__main__":
    symbols = ['GER40.cash', 'US30.cash']
    all_results = {}

    for symbol in symbols:
        print(f"\nTesting Trader Tom's strategy on {symbol}...")
        data = get_historical_data(symbol)
        if data is not None:
            pf, total_profit = trader_tom_strategy(data, symbol)
            print(f"Data period: {data.index.min()} to {data.index.max()}")
            print(f"Total trades: {pf.trades.count()}")
            print(f"Sharpe Ratio: {pf.sharpe_ratio():.4f}")
            print(f"Total Return: {pf.total_return() * 100:.2f}%")
            print(f"Max Drawdown: {pf.max_drawdown() * 100:.2f}%")
            print(f"Win Rate: {pf.trades.win_rate() * 100:.2f}%")
            all_results[symbol] = (pf, total_profit)

    print("\n============ DETAILED PERFORMANCE REPORT ============\n")
    for symbol, (pf, total_profit) in all_results.items():
        print(f"{symbol} - Trader Tom's Strategy Performance:")
        print(f"Total Return: €{total_profit:.2f}")
        print(f"Average Monthly Return: €{total_profit / 12:.2f} (over 1 year)")