#!/usr/bin/env python
# File: TradingProjectAnalyzer.py
"""
Project Summary Generator for Vectorized Trading Systems
======================================================

This script analyzes a Python project structure to generate a comprehensive
markdown summary of its components, architecture, and features for a vectorized
trading system using MT5 connectivity.

Usage:
    python TradingProjectAnalyzer.py [--output OUTPUT_FILE] [--path PROJECT_PATH] [--quiet] [--visualize]

Options:
    --output OUTPUT_FILE    Output file for the summary (default: TradingProjectAnalysis.md)
    --path PROJECT_PATH     Path to the project to analyze (default: current directory)
    --quiet                 Suppress non-critical error messages
    --visualize             Generate a dependency graph visualization (requires graphviz)
"""

import argparse
import datetime
import glob
import importlib.util
import inspect
import json
import os
import re
import sys
from collections import defaultdict

import yaml

# Global variables
QUIET_MODE = False


def load_module_from_path(file_path):
    """Dynamically load a Python module from a file path"""
    try:
        module_name = os.path.splitext(os.path.basename(file_path))[0]
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        if not QUIET_MODE:
            print(f"Error loading module {file_path}: {e}")
        return None


def extract_trading_pairs(file_paths):
    """Extract trading pairs mentioned in the codebase"""
    pairs = {
        'forex': set(),
        'crypto': set(),
        'stocks': set(),
        'indices': set()
    }

    forex_pairs = re.compile(r'(EUR|USD|GBP|JPY|AUD|NZD|CAD|CHF)(EUR|USD|GBP|JPY|AUD|NZD|CAD|CHF)')
    crypto_pairs = re.compile(r'(BTC|ETH|XRP|LTC|ADA|DOT|SOL)(USD|USDT|BTC|EUR)')
    stock_pattern = re.compile(r'(AAPL|MSFT|GOOGL|AMZN|TSLA|META|NFLX)')
    indices_pattern = re.compile(r'(SPX|NDX|DJI|FTSE|DAX|NKY)')

    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

                for match in forex_pairs.finditer(content):
                    pairs['forex'].add(match.group(0))
                for match in crypto_pairs.finditer(content):
                    pairs['crypto'].add(match.group(0))
                for match in stock_pattern.finditer(content):
                    pairs['stocks'].add(match.group(0))
                for match in indices_pattern.finditer(content):
                    pairs['indices'].add(match.group(0))
        except:
            continue

    return {k: list(v) for k, v in pairs.items() if v}


def extract_code_info(file_path):
    """Extract classes, functions, and docstrings from a Python file"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except:
        if not QUIET_MODE:
            print(f"Could not read file: {file_path}")
        return {
            'file': os.path.basename(file_path),
            'path': file_path,
            'classes': [],
            'functions': [],
            'imports': [],
            'docstring': "(File could not be read)"
        }

    info = {
        'file': os.path.basename(file_path),
        'path': file_path,
        'classes': [],
        'functions': [],
        'imports': [],
        'docstring': ""
    }

    module_docstring_match = re.search(r'^"""(.*?)"""', content, re.DOTALL | re.MULTILINE)
    if module_docstring_match:
        info['docstring'] = module_docstring_match.group(1).strip()

    module = load_module_from_path(file_path)
    if module:
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ != module.__name__:
                continue
            class_info = {
                'name': name,
                'docstring': inspect.getdoc(obj) or "",
                'methods': [],
                'inheritance': [b.__name__ for b in obj.__bases__ if b.__name__ != 'object']
            }
            for method_name, method in inspect.getmembers(obj, inspect.isfunction):
                if not method_name.startswith('__') or method_name == '__init__':
                    class_info['methods'].append({
                        'name': method_name,
                        'docstring': inspect.getdoc(method) or "",
                        'signature': str(inspect.signature(method))
                    })
            info['classes'].append(class_info)

        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if obj.__module__ == module.__name__:
                info['functions'].append({
                    'name': name,
                    'docstring': inspect.getdoc(obj) or "",
                    'signature': str(inspect.signature(obj))
                })
    else:
        class_pattern = r"class\s+(\w+)(?:\((.*?)\))?:"
        for match in re.finditer(class_pattern, content):
            class_info = {
                'name': match.group(1),
                'docstring': "",
                'methods': [],
                'inheritance': [i.strip() for i in (match.group(2) or "").split(',')] if match.group(2) else []
            }
            info['classes'].append(class_info)

        func_pattern = r"def\s+(\w+)\s*\((.*?)\):"
        for match in re.finditer(func_pattern, content):
            if not re.search(r"^\s+def\s+", match.group(0)):
                info['functions'].append({
                    'name': match.group(1),
                    'docstring': "",
                    'signature': f"({match.group(2)})"
                })

    import_pattern = r"^\s*(?:from\s+(\S+)\s+)?import\s+(.+)$"
    for line in content.split('\n'):
        match = re.match(import_pattern, line)
        if match:
            module_from = match.group(1) or ""
            imports = match.group(2).split(',')
            for imp in imports:
                imp = imp.strip()
                if ' as ' in imp:
                    name, alias = imp.split(' as ')
                    info['imports'].append({'module': module_from, 'name': name.strip(), 'alias': alias.strip()})
                else:
                    info['imports'].append({'module': module_from, 'name': imp.strip(), 'alias': None})

    info['depends_on'] = [
        imp['module'] + ('.' + imp['name'] if imp['module'] and imp['name'] != '*' else '')
        for imp in info['imports'] if imp['module'] and not imp['module'].startswith('__') and not imp['module'].startswith('builtins')
    ]
    return info


def detect_asset_classes(file_paths):
    """Detect asset-class specific configurations in the codebase"""
    asset_classes = {
        'forex': {'indicators': [], 'parameters': {}, 'risk_settings': {}},
        'crypto': {'indicators': [], 'parameters': {}, 'risk_settings': {}}
    }

    forex_pattern = r"(?:forex|FX|EURUSD|GBPUSD|currency pair)"
    crypto_pattern = r"(?:crypto|cryptocurrency|BTCUSD|Bitcoin)"

    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

                if re.search(forex_pattern, content, re.IGNORECASE):
                    ema_match = re.search(r"EMA.*?periods?.*?=?\s*\[([0-9, ]+)\]", content, re.IGNORECASE | re.DOTALL)
                    if ema_match:
                        asset_classes['forex']['parameters']['ema_periods'] = [int(x.strip()) for x in ema_match.group(1).split(',')]

                    ema_vars = re.findall(r"(ema\d+)_period\s*=\s*(\d+)", content, re.IGNORECASE)
                    if ema_vars and not asset_classes['forex']['parameters'].get('ema_periods'):
                        asset_classes['forex']['parameters']['ema_periods'] = sorted([int(p) for _, p in ema_vars])

                    rsi_match = re.search(r"RSI.*?period.*?=.*?([0-9]+)", content, re.IGNORECASE | re.DOTALL)
                    if rsi_match:
                        asset_classes['forex']['parameters']['rsi_period'] = int(rsi_match.group(1))

                    risk_match = re.search(r"risk.*?percentage.*?=.*?([0-9.]+)", content, re.IGNORECASE | re.DOTALL)
                    if risk_match:
                        asset_classes['forex']['risk_settings']['risk_percentage'] = float(risk_match.group(1))

                if re.search(crypto_pattern, content, re.IGNORECASE):
                    ema_match = re.search(r"EMA.*?periods?.*?=?\s*\[([0-9, ]+)\]", content, re.IGNORECASE | re.DOTALL)
                    if ema_match:
                        asset_classes['crypto']['parameters']['ema_periods'] = [int(x.strip()) for x in ema_match.group(1).split(',')]

                    ema_vars = re.findall(r"(ema\d+)_period\s*=\s*(\d+)", content, re.IGNORECASE)
                    if ema_vars and not asset_classes['crypto']['parameters'].get('ema_periods'):
                        asset_classes['crypto']['parameters']['ema_periods'] = sorted([int(p) for _, p in ema_vars])

                    rsi_match = re.search(r"RSI.*?period.*?=.*?([0-9]+)", content, re.IGNORECASE | re.DOTALL)
                    if rsi_match:
                        asset_classes['crypto']['parameters']['rsi_period'] = int(rsi_match.group(1))

                    risk_match = re.search(r"risk.*?percentage.*?=.*?([0-9.]+)", content, re.IGNORECASE | re.DOTALL)
                    if risk_match:
                        asset_classes['crypto']['risk_settings']['risk_percentage'] = float(risk_match.group(1))

                    vol_match = re.search(r"volatility.*?factor.*?=.*?([0-9.]+)", content, re.IGNORECASE | re.DOTALL)
                    if vol_match:
                        asset_classes['crypto']['risk_settings']['volatility_factor'] = float(vol_match.group(1))
        except:
            continue

    asset_classes['forex']['parameters'].setdefault('ema_periods', [20, 50, 200])
    asset_classes['forex']['parameters'].setdefault('rsi_period', 14)
    asset_classes['forex']['risk_settings'].setdefault('risk_percentage', 1.0)
    asset_classes['crypto']['parameters'].setdefault('ema_periods', [20, 50, 200])
    asset_classes['crypto']['parameters'].setdefault('rsi_period', 14)
    asset_classes['crypto']['risk_settings'].setdefault('risk_percentage', 1.0)
    asset_classes['crypto']['risk_settings'].setdefault('volatility_factor', 1.5)

    return asset_classes


def extract_vectorized_usage(file_paths):
    """Extract vectorized-specific configuration and usage (pandas/numpy/MT5)"""
    vector_info = {
        'indicators': set(),
        'libraries': set(),
        'mt5_usage': [],
        'optimization': []
    }

    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

                indicators = re.findall(r"(EMA|RSI|StdDev|ATR|MovingAverage)", content)
                vector_info['indicators'].update(indicators)

                if "pandas" in content:
                    vector_info['libraries'].add("pandas")
                if "numpy" in content:
                    vector_info['libraries'].add("numpy")
                if "matplotlib" in content:
                    vector_info['libraries'].add("matplotlib")

                mt5_calls = re.findall(r"mt5\.(copy_rates_from_pos|symbol_info_tick|order_send|initialize)", content)
                if mt5_calls:
                    vector_info['mt5_usage'].extend(mt5_calls)

                optimize = re.findall(r"(optimize|walk_forward|monte_carlo|parameter_opt)", content, re.IGNORECASE)
                if optimize:
                    vector_info['optimization'].extend(optimize)
        except:
            continue

    vector_info['indicators'] = sorted(list(vector_info['indicators']))
    vector_info['libraries'] = sorted(list(vector_info['libraries']))
    return vector_info


def extract_strategy_performance(file_paths):
    """Extract performance metrics from logs or results"""
    performance_data = {
        'strategies': {},
        'symbols': {},
        'timeframes': {}
    }

    log_files = [f for f in file_paths if f.endswith(('.log', '_results.txt', '_results.csv'))]

    patterns = {
        'win_rate': re.compile(r'Win(?:\s+)?(?:Rate|Percentage)(?:\s+)?(?::|=)\s*([0-9.]+)%'),
        'profit_factor': re.compile(r'Profit(?:\s+)?Factor(?:\s+)?(?::|=)\s*([0-9.]+)'),
        'sharpe': re.compile(r'Sharpe(?:\s+)?Ratio(?:\s+)?(?::|=)\s*([0-9.-]+)'),
        'max_dd': re.compile(r'Max(?:imum)?(?:\s+)?Drawdown(?:\s+)?(?::|=)\s*([0-9.-]+)%'),
        'strategy': re.compile(r'Strategy(?:\s+)?(?::|=)\s*(\w+)'),
        'symbol': re.compile(r'Symbol(?:\s+)?(?::|=)\s*(\w+)'),
        'timeframe': re.compile(r'Timeframe(?:\s+)?(?::|=)\s*(\w+)')
    }

    for file_path in log_files:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

                strategy_name = patterns['strategy'].search(content).group(1) if patterns['strategy'].search(content) else "Unknown"
                symbol = patterns['symbol'].search(content).group(1) if patterns['symbol'].search(content) else "Unknown"
                timeframe = patterns['timeframe'].search(content).group(1) if patterns['timeframe'].search(content) else "Unknown"

                metrics = {}
                for key, pattern in patterns.items():
                    match = pattern.search(content)
                    if match and key in ['win_rate', 'profit_factor', 'sharpe', 'max_dd']:
                        metrics[key] = float(match.group(1))

                if metrics:
                    for category, key in [('strategies', strategy_name), ('symbols', symbol), ('timeframes', timeframe)]:
                        if key not in performance_data[category]:
                            performance_data[category][key] = []
                        performance_data[category][key].append({
                            'symbol': symbol,
                            'timeframe': timeframe,
                            'metrics': metrics,
                            'source': os.path.basename(file_path)
                        } if category == 'strategies' else {
                            'strategy': strategy_name,
                            'timeframe': timeframe,
                            'metrics': metrics,
                            'source': os.path.basename(file_path)
                        } if category == 'symbols' else {
                            'strategy': strategy_name,
                            'symbol': symbol,
                            'metrics': metrics,
                            'source': os.path.basename(file_path)
                        })
        except:
            continue

    return performance_data


def extract_config_files(project_path):
    """Extract configuration from JSON, YAML, etc."""
    config_data = {}
    config_extensions = ['json', 'yaml', 'yml', 'ini', 'cfg']
    config_files = []
    for ext in config_extensions:
        config_files.extend(glob.glob(os.path.join(project_path, f"**/*.{ext}"), recursive=True))

    for file_path in config_files:
        try:
            config_name = os.path.basename(file_path)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                if file_path.endswith('.json'):
                    config_data[config_name] = json.load(f)
                elif file_path.endswith(('.yaml', '.yml')):
                    config_data[config_name] = yaml.safe_load(f)
                else:
                    config_data[config_name] = {}
                    current_section = 'DEFAULT'
                    for line in f:
                        line = line.strip()
                        if line.startswith('[') and line.endswith(']'):
                            current_section = line[1:-1]
                            config_data[config_name][current_section] = {}
                        elif '=' in line and not line.startswith('#'):
                            key, value = line.split('=', 1)
                            config_data[config_name].setdefault(current_section, {})[key.strip()] = value.strip()
        except:
            continue

    return config_data


def generate_dependency_graph(modules, output_file):
    """Generate a dependency graph visualization using Graphviz"""
    try:
        import graphviz
        dot = graphviz.Digraph(format='png', filename=output_file)
        dot.attr(rankdir='LR', size='11,8', ratio='fill', fontsize='10')

        for module in modules:
            module_name = os.path.splitext(module['file'])[0]
            if module_name.startswith('__'):
                continue
            shape = 'box'
            if any(c['name'].endswith('Strategy') for c in module['classes']):
                shape = 'ellipse'
            elif any(c['name'] == 'RiskManager' for c in module['classes']):
                shape = 'diamond'
            dot.node(module_name, shape=shape)

        for module in modules:
            module_name = os.path.splitext(module['file'])[0]
            if module_name.startswith('__'):
                continue
            for dep in module.get('depends_on', []):
                dep_name = dep.split('.')[0]
                if any(os.path.splitext(m['file'])[0] == dep_name for m in modules):
                    dot.edge(module_name, dep_name)

        dot.render(output_file, view=False, cleanup=True)
        return output_file + '.png'
    except Exception as e:
        print(f"Error generating dependency graph: {e}")
        return None


def analyze_project(project_path, visualize=False):
    """Analyze the entire project structure"""
    print(f"Analyzing project in: {project_path}...")
    project_info = {
        'name': os.path.basename(os.path.abspath(project_path)),
        'path': project_path,
        'modules': [],
        'packages': [],
        'recent_changes': [],
        'asset_classes': {},
        'trading_pairs': {},
        'examples': [],
        'vectorized_usage': {},
        'performance': {},
        'config_files': {},
        'dependency_graph': None
    }

    python_files = glob.glob(os.path.join(project_path, "**/*.py"), recursive=True)
    markdown_files = glob.glob(os.path.join(project_path, "**/*.md"), recursive=True)

    current_time = datetime.datetime.now()
    for file_path in python_files:
        mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
        if (current_time - mod_time).days < 7:
            project_info['recent_changes'].append(f"{os.path.relpath(file_path, project_path)} (modified on {mod_time.strftime('%Y-%m-%d')})")

    for file_path in python_files:
        project_info['modules'].append(extract_code_info(file_path))

    packages = defaultdict(list)
    for module in project_info['modules']:
        packages[os.path.dirname(os.path.relpath(module['path'], project_path))].append(module)
    project_info['packages'] = [{'name': name or 'root', 'modules': mods} for name, mods in packages.items()]

    project_info['trading_pairs'] = extract_trading_pairs(python_files)
    project_info['asset_classes'] = detect_asset_classes(python_files)
    project_info['examples'] = extract_example_commands(markdown_files + python_files)
    project_info['vectorized_usage'] = extract_vectorized_usage(python_files)
    project_info['performance'] = extract_strategy_performance(python_files + glob.glob(os.path.join(project_path, "**/*.log"), recursive=True))
    project_info['config_files'] = extract_config_files(project_path)

    if visualize:
        graph_file = os.path.join(project_path, f"{project_info['name']}_TradingDependencies")
        project_info['dependency_graph'] = generate_dependency_graph(project_info['modules'], graph_file)

    return project_info


def extract_example_commands(file_paths):
    """Extract example commands from the codebase"""
    examples = []
    for file_path in file_paths:
        if os.path.basename(file_path) in ("README.md", "USAGE.md"):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    code_blocks = re.finditer(r"```(?:python|bash)?\s*(.*?)```", content, re.DOTALL)
                    for block in code_blocks:
                        code = block.group(1).strip()
                        if "run" in code or "python" in code:
                            examples.append({'command': code, 'description': "Example command"})
            except:
                continue

    if not examples:
        examples = [
            {'description': 'Forex backtest', 'command': 'python run.py --symbol EURUSD --timeframe M15 --capital 10000'},
            {'description': 'Crypto live trading', 'command': 'python run.py --symbol BTCUSD --timeframe M15 --live'},
            {'description': 'Parameter optimization', 'command': 'python run.py --symbol DAX --timeframe M15 --optimize'}
        ]
    return examples


def generate_summary_markdown(project_info, output_file):
    """Generate a markdown summary from the project information"""
    print(f"Generating summary to: {output_file}...")
    summary = "# " + project_info['name'] + " - Vectorized Trading System Summary\n\n"
    summary += "Generated on: " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n\n"

    summary += "## Core Components\n\n"
    core_modules = [m for m in project_info['modules'] if os.path.splitext(m['file'])[0] in
                    ["engine", "strategy", "risk", "parameters", "run"]]
    for module in sorted(core_modules, key=lambda x: x['file']):
        rel_path = os.path.relpath(module['path'], project_info['path'])
        summary += "### " + module['file'] + " - " + os.path.dirname(rel_path) + "\n\n"
        if module['docstring']:
            summary += module['docstring'] + "\n\n"

        if module['classes']:
            summary += "**Classes:**\n"
            for cls in module['classes']:
                inheritance = " (inherits from " + ', '.join(cls['inheritance']) + ")" if cls['inheritance'] else ""
                summary += "- **" + cls['name'] + inheritance + "**: " + (cls['docstring'].split('.')[0] if cls['docstring'] else '') + "\n"
                for method in cls['methods'][:3]:
                    summary += "    - **" + method['name'] + "**: " + (method['docstring'].split('.')[0] if method['docstring'] else method['signature']) + "\n"
            summary += "\n"

        if module['functions']:
            summary += "**Main Functions:**\n"
            for func in module['functions']:
                summary += "- **" + func['name'] + "**: " + (func['docstring'].split('.')[0] if func['docstring'] else '') + "\n"
            summary += "\n"

    if project_info['recent_changes']:
        summary += "## Recent Changes\n"
        for change in project_info['recent_changes'][:5]:
            summary += "- " + change + "\n"
        summary += "\n"

    summary += "## Asset Class Support\n"
    for asset_name, asset_info in project_info['asset_classes'].items():
        summary += "- **" + asset_name.capitalize() + "**: "
        parts = []
        if 'ema_periods' in asset_info['parameters']:
            parts.append("EMA periods=" + '/'.join(map(str, asset_info['parameters']['ema_periods'])))
        if 'rsi_period' in asset_info['parameters']:
            parts.append("RSI=" + str(asset_info['parameters']['rsi_period']))
        if 'risk_percentage' in asset_info['risk_settings']:
            parts.append("risk=" + str(asset_info['risk_settings']['risk_percentage']) + "%")
        if 'volatility_factor' in asset_info['risk_settings']:
            parts.append("volatility factor=" + str(asset_info['risk_settings']['volatility_factor']))
        summary += ", ".join(parts) + "\n"
    summary += "\n"

    if project_info['trading_pairs']:
        summary += "## Supported Trading Instruments\n"
        for asset_type, pairs in project_info['trading_pairs'].items():
            if pairs:
                summary += "- **" + asset_type.capitalize() + "**: " + ', '.join(pairs) + "\n"
        summary += "\n"

    vector_usage = project_info['vectorized_usage']
    if vector_usage:
        summary += "## Vectorized Implementation Details\n"
        if vector_usage['indicators']:
            summary += "**Indicators Used:**\n" + "\n".join(["- " + i for i in vector_usage['indicators']]) + "\n\n"
        if vector_usage['libraries']:
            summary += "**Libraries:**\n" + "\n".join(["- " + l for l in vector_usage['libraries']]) + "\n\n"
        if vector_usage['mt5_usage']:
            summary += "**MT5 Integration:**\n" + "\n".join(["- " + m for m in vector_usage['mt5_usage']]) + "\n\n"
        if vector_usage['optimization']:
            summary += "**Optimization Features:**\n" + "\n".join(["- " + o for o in vector_usage['optimization']]) + "\n\n"

    if project_info['performance']['strategies']:
        summary += "## Strategy Performance\n"
        for strategy_name, results in project_info['performance']['strategies'].items():
            summary += "### " + strategy_name + "\n"
            for result in results:
                summary += "**" + result['symbol'] + " (" + result['timeframe'] + "):**\n"
                for metric, value in result['metrics'].items():
                    unit = "%" if metric in ['win_rate', 'max_dd'] else ""
                    summary += "- " + metric.replace('_', ' ').title() + ": " + f"{value:.2f}" + unit + "\n"
                summary += "- Source: " + result['source'] + "\n\n"

    if project_info['config_files']:
        summary += "## Configuration Files\n"
        summary += "\n".join(["- " + name for name in project_info['config_files'].keys()]) + "\n\n"

    if project_info['dependency_graph']:
        summary += "## Module Dependencies\n"
        summary += "- [" + os.path.basename(project_info['dependency_graph']) + "](" + project_info['dependency_graph'] + ")\n\n"

    summary += "## Usage Examples\n"
    for i, example in enumerate(project_info['examples'][:3], 1):
        summary += str(i) + ". " + example['description'] + ":\n   ```\n   " + example['command'] + "\n   ```\n"

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(summary)
    return summary


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate a summary of a vectorized Python trading system project')
    parser.add_argument('--path', default='.', help='Path to the project directory')
    parser.add_argument('--output', default='TradingProjectAnalysis.md', help='Output file for the summary')
    parser.add_argument('--quiet', action='store_true', help='Suppress non-critical error messages')
    parser.add_argument('--visualize', action='store_true', help='Generate a dependency graph (requires graphviz)')
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    global QUIET_MODE
    QUIET_MODE = args.quiet

    project_info = analyze_project(args.path, args.visualize)
    summary = generate_summary_markdown(project_info, args.output)

    print(f"Project summary written to {args.output}")
    print(f"Found {len(project_info['modules'])} Python modules")
    print(f"Identified {len(project_info['recent_changes'])} recent changes")
    print("Generated " + str(summary.count('\n') + 1) + " lines of documentation")
    if args.visualize and project_info['dependency_graph']:
        print(f"Dependency graph generated at: {project_info['dependency_graph']}")


if __name__ == "__main__":
    main()