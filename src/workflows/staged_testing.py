# src/workflows/staged_testing.py
"""
Sophy3 - Staged Testing Script
Functie: Systematische, gefaseerde backtesting van trading strategieën
Auteur: AI Trading Assistant
Laatste update: 2025-04-02

Gebruik:
  python src/workflows/staged_testing.py

Dependencies:
  - pandas
  - vectorbt
  - data.cache
  - gc (garbage collection)
  - psutil (geheugenmonitoring)
"""

import os
import sys
import pandas as pd
import vectorbt as vbt
import time
import logging
import gc
import psutil
from datetime import datetime, timedelta
import argparse

# Voeg de root directory toe aan sys.path zodat modules gevonden worden
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
print(f"[INFO] Project root toegevoegd aan sys.path: {root_dir}")

try:
    from src.strategies.ema_strategy import multi_layer_ema_strategy
    from src.strategies.params import get_strategy_params, get_risk_params, detect_asset_class
    from src.data import get_data, initialize_mt5, shutdown_mt5
    print("[INFO] Alle benodigde modules zijn succesvol geïmporteerd")
except ImportError as e:
    print(f"[ERROR] Importfout: {e}")
    print(f"[INFO] Huidige sys.path: {sys.path}")
    sys.exit(1)

# Stel logger in
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    handlers=[logging.FileHandler("staged_testing.log"),
                              logging.StreamHandler()])
logger = logging.getLogger(__name__)

def log_memory_usage():
    """Log het huidige geheugengebruik."""
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / (1024 * 1024)
    logger.info(f"Geheugengebruik: {memory_mb:.2f} MB")
    return memory_mb

def filter_strategy_params(strategy_params):
    """
    Filtert parameters om alleen de relevante door te geven aan de strategie functie.
    """
    valid_params = ['ema_periods', 'rsi_period', 'rsi_oversold', 'rsi_overbought', 'volatility_factor']
    return {k: v for k, v in strategy_params.items() if k in valid_params}

def convert_timeframe_to_pandas_freq(timeframe):
    """
    Converteer custom timeframe formaat naar pandas frequency formaat.
    """
    if timeframe.startswith('M'):
        minutes = timeframe[1:]
        return f"{minutes}T"
    elif timeframe.startswith('H'):
        hours = timeframe[1:]
        return f"{hours}h"
    elif timeframe.startswith('D'):
        days = timeframe[1:]
        return f"{days}D"
    elif timeframe.startswith('W'):
        weeks = timeframe[1:]
        return f"{weeks}W"
    logger.warning(f"Onbekend timeframe-formaat: {timeframe}, default naar '1D'")
    return "1D"

def run_backtest(symbol, timeframe, lookback_days=None, lookback_bars=None):
    """
    Voert een backtest uit voor een specifiek symbool en timeframe.
    """
    start_time = time.time()

    if lookback_days is not None:
        if timeframe == 'M5':
            lookback_bars = lookback_days * 24 * 12
        elif timeframe == 'M15':
            lookback_bars = lookback_days * 24 * 4
        elif timeframe == 'H1':
            lookback_bars = lookback_days * 24
        elif timeframe == 'H4':
            lookback_bars = lookback_days * 6
        elif timeframe == 'D1':
            lookback_bars = lookback_days

    if lookback_bars is None:
        lookback_bars = {'M5': 1000, 'M15': 1000, 'H1': 1500, 'H4': 1000, 'D1': 500}.get(timeframe, 500)

    df = get_data(symbol, timeframe, bars=lookback_bars)
    if df is None or len(df) == 0:
        logger.warning(f"Geen data beschikbaar voor {symbol} {timeframe}, overslaan")
        return None

    strategy_params = get_strategy_params(symbol)
    risk_params = get_risk_params(symbol)

    ema_params = filter_strategy_params(strategy_params)
    entries, exits = multi_layer_ema_strategy(df, **ema_params)

    pandas_freq = convert_timeframe_to_pandas_freq(timeframe)

    try:
        portfolio = vbt.Portfolio.from_signals(close=df['close'], entries=entries,
                                               exits=exits, size=1.0, freq=pandas_freq)

        total_return = portfolio.total_return()
        sharpe_ratio = portfolio.sharpe_ratio()
        max_drawdown = portfolio.max_drawdown()

        trades = portfolio.trades.records
        trades_df = trades.to_pandas() if hasattr(trades, "to_pandas") else pd.DataFrame(trades)
        win_rate = len(trades_df[trades_df['return'] > 0]) / len(trades_df) * 100 if len(trades_df) > 0 else 0
        avg_trade = trades_df['return'].mean() if len(trades_df) > 0 else 0

        gross_profit = trades_df[trades_df['return'] > 0]['return'].sum() if len(trades_df[trades_df['return'] > 0]) > 0 else 0
        gross_loss = abs(trades_df[trades_df['return'] < 0]['return'].sum()) if len(trades_df[trades_df['return'] < 0]) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        elapsed_time = time.time() - start_time

        results = {'symbol': symbol, 'timeframe': timeframe,
                   'total_return': total_return, 'sharpe_ratio': sharpe_ratio,
                   'max_drawdown': max_drawdown, 'win_rate': win_rate, 'avg_trade': avg_trade,
                   'profit_factor': profit_factor, 'trade_count': len(trades),
                   'test_period': f"{df.index[0]} tot {df.index[-1]}", 'bars_count': len(df),
                   'runtime_seconds': elapsed_time}

        logger.info(f"Backtest voltooid: {symbol} {timeframe} - Sharpe: {sharpe_ratio:.2f}, Return: {total_return:.2%}")
        return results

    except Exception as e:
        logger.error(f"Fout bij backtest {symbol} {timeframe}: {str(e)}")
        return None
    finally:
        portfolio = None
        df = None
        gc.collect()
        log_memory_usage()

def run_asset_class_screening(timeframe='H1'):
    """
    Fase 1: Screeningtest van verschillende asset classes.
    """
    logger.info(f"Fase 1: Asset Class Screening op {timeframe}")
    print(f"\nFASE 1: ASSET CLASS SCREENING OP {timeframe}")
    print("=" * 70)

    representatives = {'forex': ['EURUSD', 'GBPUSD'], 'crypto': ['BTCUSD', 'ETHUSD'],
                      'stocks': ['AAPL', 'MSFT'], 'indices': ['SPX', 'NDX']}

    results = []

    for asset_class, symbols in representatives.items():
        print(f"\nTesting {asset_class.upper()} asset class...")
        for symbol in symbols:
            print(f"  Testing {symbol}...")
            result = run_backtest(symbol, timeframe)
            if result:
                result['asset_class'] = asset_class
                results.append(result)
                print(f"  VOLTOOID: {symbol}: Sharpe={result['sharpe_ratio']:.2f}, Return={result['total_return']:.2%}")
            else:
                print(f"  MISLUKT: {symbol}: Test mislukt")

    if results:
        results_df = pd.DataFrame(results)
        os.makedirs('results', exist_ok=True)
        avg_by_class = results_df.groupby('asset_class')[['sharpe_ratio', 'total_return', 'max_drawdown', 'win_rate']].mean()

        print("\n" + "=" * 70)
        print("ASSET CLASS PERFORMANCE SAMENVATTING")
        print("=" * 70)
        print(avg_by_class.to_string())
        print("\n" + "=" * 70)
        print("TOP PERFORMERS")
        print("=" * 70)
        print(results_df.sort_values('sharpe_ratio', ascending=False).to_string(index=False))

        results_df.to_csv(f'results/phase1_asset_screening_{datetime.now().strftime("%Y%m%d_%H%M")}.csv', index=False)

        best_asset_class = avg_by_class['sharpe_ratio'].idxmax()
        best_instruments = results_df[results_df['asset_class'] == best_asset_class].sort_values('sharpe_ratio', ascending=False)['symbol'].tolist()

        print(f"\nBeste asset class: {best_asset_class.upper()}")
        print(f"Beste instrumenten: {', '.join(best_instruments[:2])}")
        return best_asset_class, best_instruments[:2]

    logger.warning("Geen resultaten van asset class screening")
    return None, None

def run_timeframe_optimization(symbols, timeframes=['M5', 'M15', 'H1', 'H4', 'D1']):
    """
    Fase 2: Timeframe optimalisatie.
    """
    logger.info(f"Fase 2: Timeframe Optimalisatie voor {symbols}")
    print(f"\nFASE 2: TIMEFRAME OPTIMALISATIE")
    print("=" * 70)

    results = []

    for symbol in symbols:
        print(f"\nTesting {symbol} op verschillende timeframes...")
        for timeframe in timeframes:
            print(f"  Testing {timeframe}...")
            lookback_days = {'M5': 14, 'M15': 30, 'H1': 60, 'H4': 120, 'D1': 365}.get(timeframe, 60)
            result = run_backtest(symbol, timeframe, lookback_days=lookback_days)
            if result:
                results.append(result)
                print(f"  VOLTOOID: {timeframe}: Sharpe={result['sharpe_ratio']:.2f}, Return={result['total_return']:.2%}")
            else:
                print(f"  MISLUKT: {timeframe}: Test mislukt")

    if results:
        results_df = pd.DataFrame(results)
        avg_by_timeframe = results_df.groupby('timeframe')[['sharpe_ratio', 'total_return', 'max_drawdown', 'win_rate']].mean()

        print("\n" + "=" * 70)
        print("TIMEFRAME PERFORMANCE SAMENVATTING")
        print("=" * 70)
        print(avg_by_timeframe.to_string())
        print("\n" + "=" * 70)
        print("TOP TIMEFRAME COMBINATIES")
        print("=" * 70)
        print(results_df.sort_values('sharpe_ratio', ascending=False).to_string(index=False))

        results_df.to_csv(f'results/phase2_timeframe_optimization_{datetime.now().strftime("%Y%m%d_%H%M")}.csv', index=False)

        best_timeframe = avg_by_timeframe['sharpe_ratio'].idxmax()
        best_combinations = results_df.sort_values('sharpe_ratio', ascending=False)[['symbol', 'timeframe']].head(3).values.tolist()

        print(f"\nBeste timeframe overall: {best_timeframe}")
        print(f"Top 3 symbol-timeframe combinaties:")
        for i, (symbol, timeframe) in enumerate(best_combinations, 1):
            print(f"  {i}. {symbol} op {timeframe}")
        return best_timeframe, best_combinations

    logger.warning("Geen resultaten van timeframe optimalisatie")
    return None, None

def run_instrument_diversification(asset_class, timeframe):
    """
    Fase 3: Instrument diversificatie.
    """
    logger.info(f"Fase 3: Instrument Diversificatie voor {asset_class} op {timeframe}")
    print(f"\nFASE 3: INSTRUMENT DIVERSIFICATIE - {asset_class.upper()} OP {timeframe}")
    print("=" * 70)

    instruments = {
        'forex': ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'EURGBP', 'EURCHF', 'EURJPY'],
        'crypto': ['BTCUSD', 'ETHUSD', 'XRPUSD', 'ADAUSD', 'DOTUSD', 'LTCUSD', 'SOLUSD', 'BNBUSD'],
        'stocks': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'V', 'JNJ'],
        'indices': ['SPX', 'NDX', 'DJI', 'DAX', 'FTSE', 'NKY', 'HSI', 'STOXX50E']
    }

    if asset_class not in instruments:
        logger.error(f"Onbekende asset class: {asset_class}")
        return None

    symbols = instruments[asset_class]
    results = []

    print(f"Testing {len(symbols)} instrumenten...")
    for symbol in symbols:
        print(f"  Testing {symbol}...")
        result = run_backtest(symbol, timeframe)
        if result:
            results.append(result)
            print(f"  VOLTOOID: {symbol}: Sharpe={result['sharpe_ratio']:.2f}, Return={result['total_return']:.2%}")
        else:
            print(f"  MISLUKT: {symbol}: Test mislukt")

    if results:
        results_df = pd.DataFrame(results)
        print("\n" + "=" * 70)
        print(f"INSTRUMENT PERFORMANCE BINNEN {asset_class.upper()}")
        print("=" * 70)
        print(results_df.sort_values('sharpe_ratio', ascending=False).to_string(index=False))

        results_df.to_csv(f'results/phase3_instrument_diversification_{asset_class}_{datetime.now().strftime("%Y%m%d_%H%M")}.csv', index=False)

        best_instruments = results_df.sort_values('sharpe_ratio', ascending=False)['symbol'].head(3).tolist()
        print(f"\nTop 3 instrumenten in {asset_class}:")
        for i, symbol in enumerate(best_instruments, 1):
            idx = results_df[results_df['symbol'] == symbol].index[0]
            print(f"  {i}. {symbol} - Sharpe: {results_df.loc[idx, 'sharpe_ratio']:.2f}, Return: {results_df.loc[idx, 'total_return']:.2%}")
        return best_instruments

    logger.warning("Geen resultaten van instrument diversificatie")
    return None

def run_parameter_optimization(symbol, timeframe):
    """
    Fase 4: Parameter optimalisatie.
    """
    logger.info(f"Fase 4: Parameter Optimalisatie voor {symbol} op {timeframe}")
    print(f"\nFASE 4: PARAMETER OPTIMALISATIE - {symbol} OP {timeframe}")
    print("=" * 70)

    try:
        from src.backtesting.backtest import optimize_parameters, run_backtest as backtest_run

        log_memory_usage()
        print(f"Data ophalen voor {symbol} {timeframe}...")
        lookback_days = {'M5': 14, 'M15': 30, 'H1': 60, 'H4': 120, 'D1': 365}.get(timeframe, 365)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        df = get_data(symbol, timeframe, start_date=start_date, end_date=end_date)
        if df is None or len(df) == 0:
            logger.error(f"Geen data beschikbaar voor {symbol} {timeframe}")
            return None

        strategy_params = get_strategy_params(symbol)
        risk_params = get_risk_params(symbol)

        print(f"Optimaliseren van EMA parameters...")
        ema_params, best_ema, _ = optimize_parameters(df, symbol, strategy_params,
                                                      risk_params, param_to_optimize='ema',
                                                      optimization_metric='sharpe', initial_capital=10000.0)

        print(f"Optimaliseren van RSI parameters...")
        rsi_params, best_rsi, _ = optimize_parameters(df, symbol, ema_params,
                                                      risk_params, param_to_optimize='rsi',
                                                      optimization_metric='sharpe', initial_capital=10000.0)

        print(f"Optimaliseren van volatiliteit parameters...")
        final_params, best_vol, _ = optimize_parameters(df, symbol, rsi_params,
                                                        risk_params, param_to_optimize='volatility',
                                                        optimization_metric='sharpe', initial_capital=10000.0)

        print(f"Uitvoeren van finale backtest met geoptimaliseerde parameters...")
        results, portfolio = backtest_run(df, final_params, risk_params, 10000.0, True)

        print("\n" + "=" * 70)
        print("OPTIMALE PARAMETERS")
        print("=" * 70)
        print(f"EMA periodes: {final_params['ema_periods']}")
        print(f"RSI periode: {final_params['rsi_period']}")
        print(f"RSI oversold/overbought: {final_params['rsi_oversold']}/{final_params['rsi_overbought']}")
        print(f"Volatiliteitsfactor: {final_params['volatility_factor']}")

        print("\n" + "=" * 70)
        print("PRESTATIEMETRIEKEN MET OPTIMALE PARAMETERS")
        print("=" * 70)
        for metric, value in results.items():
            if isinstance(value, (int, float)):
                if metric in ['total_return', 'max_drawdown']:
                    print(f"{metric}: {value:.2%}")
                else:
                    print(f"{metric}: {value:.4f}")

        with open(f'results/phase4_optimal_params_{symbol}_{timeframe}_{datetime.now().strftime("%Y%m%d_%H%M")}.txt', 'w') as f:
            f.write(f"Symbol: {symbol}\n")
            f.write(f"Timeframe: {timeframe}\n\n")
            f.write("Optimal Parameters:\n")
            f.write(f"EMA periods: {final_params['ema_periods']}\n")
            f.write(f"RSI period: {final_params['rsi_period']}\n")
            f.write(f"RSI oversold/overbought: {final_params['rsi_oversold']}/{final_params['rsi_overbought']}\n")
            f.write(f"Volatility factor: {final_params['volatility_factor']}\n\n")
            f.write("Performance:\n")
            for metric, value in results.items():
                if isinstance(value, (int, float)):
                    f.write(f"{metric}: {value}\n")

        if portfolio and hasattr(portfolio, "stats"):
            try:
                portfolio.stats().to_csv(f'results/phase4_portfolio_stats_{symbol}_{timeframe}_{datetime.now().strftime("%Y%m%d_%H%M")}.csv')
            except:
                pass

        return final_params, results

    except Exception as e:
        logger.error(f"Fout bij parameter optimalisatie: {str(e)}")
        print(f"FOUT: {str(e)}")
        return None
    finally:
        df = None
        gc.collect()
        log_memory_usage()

def run_staged_testing(skip_to_phase=None, specific_assets=None, specific_timeframes=None):
    """
    Voer het volledige gefaseerde testproces uit, of begin bij een specifieke fase.
    """
    os.makedirs('results', exist_ok=True)

    overall_start_time = time.time()
    print("\n" + "=" * 70)
    print("SOPHY3 GEFASEERDE STRATEGIE TESTING")
    print("=" * 70)
    print(f"Start tijd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_memory_usage()

    results = {'phase1': None, 'phase2': None, 'phase3': None, 'phase4': None}

    if skip_to_phase is None or skip_to_phase <= 1:
        best_asset_class, best_instruments = run_asset_class_screening()
        results['phase1'] = (best_asset_class, best_instruments)
    else:
        best_asset_class = None
        best_instruments = specific_assets or ['EURUSD', 'BTCUSD']
        print(f"\nFase 1 overgeslagen, gebruikt instrumenten: {best_instruments}")

    if skip_to_phase is None or skip_to_phase <= 2:
        if best_instruments:
            best_timeframe, best_combinations = run_timeframe_optimization(best_instruments)
            results['phase2'] = (best_timeframe, best_combinations)
        else:
            print("\nKon fase 2 niet uitvoeren, geen instrumenten beschikbaar uit fase 1")
            best_timeframe = None
            best_combinations = []
    else:
        best_timeframe = specific_timeframes[0] if specific_timeframes else 'H1'
        best_combinations = [(instr, best_timeframe) for instr in best_instruments] if best_instruments else []
        print(f"\nFase 2 overgeslagen, gebruikt timeframe: {best_timeframe}")

    if skip_to_phase is None or skip_to_phase <= 3:
        if best_asset_class is None and best_instruments:
            best_asset_class = detect_asset_class(best_instruments[0])
        if best_asset_class and best_timeframe:
            best_diversified_instruments = run_instrument_diversification(best_asset_class, best_timeframe)
            results['phase3'] = best_diversified_instruments
        else:
            print("\nKon fase 3 niet uitvoeren door ontbrekende asset class of timeframe")
    else:
        best_diversified_instruments = [combo[0] for combo in best_combinations[:3]] if best_combinations else best_instruments
        print(f"\nFase 3 overgeslagen, gebruikt top instrumenten: {best_diversified_instruments}")

    if skip_to_phase is None or skip_to_phase <= 4:
        if best_combinations and len(best_combinations) > 0:
            top_symbol, top_timeframe = best_combinations[0]
            optimal_params, optimization_results = run_parameter_optimization(top_symbol, top_timeframe)
            results['phase4'] = (optimal_params, optimization_results)
        else:
            print("\nKon fase 4 niet uitvoeren door ontbrekende beste combinatie")

    overall_elapsed_time = time.time() - overall_start_time
    print("\n" + "=" * 70)
    print("SAMENVATTING GEFASEERDE TESTING")
    print("=" * 70)
    print(f"Totale testtijd: {overall_elapsed_time:.2f} seconden ({overall_elapsed_time / 60:.2f} minuten)")

    if results['phase1']:
        print(f"\nFase 1 - Beste asset class: {results['phase1'][0]}")
        print(f"         Beste instrumenten: {', '.join(results['phase1'][1])}")
    if results['phase2']:
        print(f"\nFase 2 - Beste timeframe: {results['phase2'][0]}")
        print(f"         Top 3 combinaties:")
        for i, (symbol, tf) in enumerate(results['phase2'][1][:3], 1):
            print(f"         {i}. {symbol} op {tf}")
    if results['phase3']:
        print(f"\nFase 3 - Beste gediversifieerde instrumenten:")
        for i, instr in enumerate(results['phase3'][:3], 1):
            print(f"         {i}. {instr}")
    if results['phase4'] and results['phase4'][0]:
        params = results['phase4'][0]
        print(f"\nFase 4 - Optimale parameters voor {top_symbol} op {top_timeframe}:")
        print(f"         EMA periodes: {params['ema_periods']}")
        print(f"         RSI periode: {params['rsi_period']}")
        print(f"         RSI oversold/overbought: {params['rsi_oversold']}/{params['rsi_overbought']}")
        print(f"         Volatiliteitsfactor: {params['volatility_factor']}")

    with open(f'results/staged_testing_summary_{datetime.now().strftime("%Y%m%d_%H%M")}.txt', 'w') as f:
        f.write(f"SOPHY3 GEFASEERDE STRATEGIE TESTING\n")
        f.write(f"Uitgevoerd op: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Totale testtijd: {overall_elapsed_time:.2f} seconden ({overall_elapsed_time / 60:.2f} minuten)\n\n")
        if results['phase1']:
            f.write(f"Fase 1 - Asset Class Screening\n")
            f.write(f"Beste asset class: {results['phase1'][0]}\n")
            f.write(f"Beste instrumenten: {', '.join(results['phase1'][1])}\n\n")
        if results['phase2']:
            f.write(f"Fase 2 - Timeframe Optimalisatie\n")
            f.write(f"Beste timeframe: {results['phase2'][0]}\n")
            f.write(f"Top 3 combinaties:\n")
            for i, (symbol, tf) in enumerate(results['phase2'][1][:3], 1):
                f.write(f"{i}. {symbol} op {tf}\n")
            f.write("\n")
        if results['phase3']:
            f.write(f"Fase 3 - Instrument Diversificatie\n")
            f.write(f"Beste gediversifieerde instrumenten:\n")
            for i, instr in enumerate(results['phase3'][:3], 1):
                f.write(f"{i}. {instr}\n")
            f.write("\n")
        if results['phase4'] and results['phase4'][0]:
            params = results['phase4'][0]
            f.write(f"Fase 4 - Parameter Optimalisatie voor {top_symbol} op {top_timeframe}\n")
            f.write(f"EMA periodes: {params['ema_periods']}\n")
            f.write(f"RSI periode: {params['rsi_period']}\n")
            f.write(f"RSI oversold/overbought: {params['rsi_oversold']}/{params['rsi_overbought']}\n")
            f.write(f"Volatiliteitsfactor: {params['volatility_factor']}\n")

    print("\n" + "=" * 70)
    print(f"Volledige testresultaten opgeslagen in de 'results' directory")
    print("=" * 70)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sophy3 Gefaseerde Trading Strategie Testing")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4], help="Begin bij specifieke fase")
    parser.add_argument("--symbols", type=str, nargs='+', help="Test specifieke symbolen (indien fase overgeslagen)")
    parser.add_argument("--timeframes", type=str, nargs='+', help="Test specifieke timeframes (indien fase overgeslagen)")

    args = parser.parse_args()
    run_staged_testing(skip_to_phase=args.phase, specific_assets=args.symbols, specific_timeframes=args.timeframes)