import pandas as pd
import numpy as np
import vectorbt as vbt

def simple_ema_strategy(df, ema_short=50, ema_long=200, atr_period=14, sl_atr=1.5, tp_atr=3.0):
    """
    Een simpele EMA crossover-strategie met ATR-gebaseerde SL/TP.
    """
    df = df.copy()
    df['ema_short'] = df['close'].ewm(span=ema_short, adjust=False).mean()
    df['ema_long'] = df['close'].ewm(span=ema_long, adjust=False).mean()
    atr = vbt.ATR.run(df['high'], df['low'], df['close'], window=atr_period).atr
    entries = df['ema_short'].gt(df['ema_long'])
    exits = df['ema_short'].lt(df['ema_long'])
    entries = entries.fillna(False)
    exits = exits.fillna(False)
    sl_pct = (atr * sl_atr) / df['close']
    tp_pct = (atr * tp_atr) / df['close']
    return entries, exits, sl_pct, tp_pct