#!/usr/bin/env python
# File: TradingProjectAnalyzer.py
"""
Project Summary Generator for Trading Systems
============================================

This script analyzes a Python trading project structure (optimized for VectorBT) to generate a comprehensive
markdown summary of its components, architecture, and features.

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
    """Dynamically load a Python module from a file path."""
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
    """Extract trading pairs mentioned in the codebase."""
    pairs = {
        'forex': set(),
        'crypto': set(),
        'stocks': set(),
        'indices': set()
    }

    forex_pairs = re.compile(r'\b([A-Z]{3})(USD|EUR|GBP|JPY|AUD|NZD|CAD|CHF)\b')
    crypto_pairs = re.compile(r'\b(BTC|ETH|XRP|LTC|ADA|DOT|SOL|BCH|XMR)(USD|USDT|BTC|ETH|EUR)\b')
    stock_pattern = re.compile(r'\b(AAPL|MSFT|GOOGL|AMZN|TSLA|META|NFLX|NVDA|JPM|BA)\b')
    indices_pattern = re.compile(r'\b(SPX|NDX|DJI|FTSE|DAX|NKY|CAC|HSI)\b')

    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

                pairs['forex'].update(forex_pairs.findall(content))
                pairs['crypto'].update(crypto_pairs.findall(content))
                pairs['stocks'].update(stock_pattern.findall(content))
                pairs['indices'].update(indices_pattern.findall(content))
        except:
            continue

    return {k: sorted(list(v)) for k, v in pairs.items() if v}

def extract_code_info(file_path):
    """Extract classes, functions, docstrings, and other metadata from a Python file."""
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
            'docstring': "(File could not be read)",
            'internal_depends_on': [],
            'has_main_block': False,
            'uses_vectorbt': False
        }

    info = {
        'file': os.path.basename(file_path),
        'path': file_path,
        'classes': [],
        'functions': [],
        'imports': [],
        'docstring': "",
        'internal_depends_on': [],
        'has_main_block': 'if __name__ == "__main__":' in content,
        'uses_vectorbt': 'vbt.Portfolio.from_signals' in content
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
                    method_info = {
                        'name': method_name,
                        'docstring': inspect.getdoc(method) or "",
                        'signature': str(inspect.signature(method))
                    }
                    class_info['methods'].append(method_info)
            info['classes'].append(class_info)

        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if obj.__module__ == module.__name__:
                func_info = {
                    'name': name,
                    'docstring': inspect.getdoc(obj) or "",
                    'signature': str(inspect.signature(obj))
                }
                info['functions'].append(func_info)

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
                    info['imports'].append({
                        'module': module_from,
                        'name': name.strip(),
                        'alias': alias.strip()
                    })
                else:
                    info['imports'].append({
                        'module': module_from,
                        'name': imp.strip(),
                        'alias': None
                    })

    info['depends_on'] = [
        imp['module'] + ('.' + imp['name'] if imp['module'] and imp['name'] != '*' else '')
        for imp in info['imports'] if imp['module'] and not imp['module'].startswith('__')
    ]

    # Detecteer interne projectafhankelijkheden
    project_root = os.path.dirname(os.path.dirname(file_path))
    internal_deps = []
    for imp in info['imports']:
        module_from = imp['module']
        if module_from.startswith('src.'):
            internal_deps.append(module_from)
    info['internal_depends_on'] = internal_deps

    return info

def detect_asset_classes(file_paths):
    """Detect asset-class specific configurations and parameters."""
    asset_classes = {
        'forex': {'indicators': [], 'parameters': {}, 'risk_settings': {}},
        'crypto': {'indicators': [], 'parameters': {}, 'risk_settings': {}},
        'stocks': {'indicators': [], 'parameters': {}, 'risk_settings': {}},
        'indices': {'indicators': [], 'parameters': {}, 'risk_settings': {}}
    }

    # Laad params.py dynamisch
    params_path = None
    for file_path in file_paths:
        if file_path.endswith('params.py'):
            params_path = file_path
            break

    if params_path:
        params_module = load_module_from_path(params_path)
        if params_module:
            for asset in asset_classes:
                # Haal parameters op
                strategy_params = params_module.get_strategy_params(asset)
                risk_params = params_module.get_risk_params(asset)
                asset_classes[asset]['parameters'] = strategy_params
                asset_classes[asset]['risk_settings'] = risk_params

    return asset_classes

def extract_vectorbt_usage(file_paths):
    """Extract VectorBT-specific configuration and usage."""
    vbt_info = {
        'strategies': [],
        'metrics': set(),
        'portfolio_settings': {}
    }

    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

                # Detect VectorBT strategies
                strat_match = re.findall(r"vbt\.Portfolio\.from_signals\((.*?)\)", content, re.DOTALL)
                if strat_match:
                    for match in strat_match:
                        vbt_info['strategies'].append({'file': os.path.basename(file_path), 'config': match.strip()})

                # Detect metrics
                metrics = re.findall(r"portfolio\.(\w+)\(\)", content)
                vbt_info['metrics'].update(metrics)

                # Detect portfolio settings
                settings = re.search(r"vbt\.Portfolio\.from_signals\((.*?)\)", content, re.DOTALL)
                if settings:
                    vbt_info['portfolio_settings'] = {
                        'file': os.path.basename(file_path),
                        'settings': settings.group(1)
                    }
        except:
            continue

    vbt_info['metrics'] = sorted(list(vbt_info['metrics']))
    return vbt_info

def extract_config_files(project_path):
    """Extract configuration from JSON and YAML files."""
    config_data = {}
    config_files = glob.glob(os.path.join(project_path, "**/*.json"), recursive=True) + \
                   glob.glob(os.path.join(project_path, "**/*.yaml"), recursive=True) + \
                   glob.glob(os.path.join(project_path, "**/*.yml"), recursive=True)

    for file_path in config_files:
        try:
            config_name = os.path.basename(file_path)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                if file_path.endswith('.json'):
                    config_data[config_name] = json.load(f)
                else:
                    config_data[config_name] = yaml.safe_load(f)
        except:
            continue
    return config_data

def generate_dependency_graph(modules, output_file):
    """Generate a dependency graph visualization using Graphviz."""
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
            elif 'risk' in module_name.lower():
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
        return f"{output_file}.png"
    except Exception as e:
        print(f"Error generating dependency graph: {e}")
        return None

def analyze_project(project_path, visualize=False):
    """Analyze the entire project structure."""
    project_info = {
        'name': os.path.basename(os.path.abspath(project_path)),
        'path': project_path,
        'modules': [],
        'packages': [],
        'recent_changes': [],
        'asset_classes': {},
        'trading_pairs': {},
        'vectorbt_usage': {},
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
        module_info = extract_code_info(file_path)
        project_info['modules'].append(module_info)

    # Bouw een afhankelijkheidsgrafiek om te bepalen welke modules "core" zijn
    dependency_counts = defaultdict(int)
    for module in project_info['modules']:
        module_name = os.path.splitext(module['file'])[0]
        for dep in module['internal_depends_on']:
            dep_name = dep.split('.')[-1]  # Bijv. 'backtest' uit 'src.backtesting.backtest'
            dependency_counts[dep_name] += 1

    # Classificeer modules
    core_modules = []
    workflow_modules = []
    for module in project_info['modules']:
        module_name = os.path.splitext(module['file'])[0]
        # Een module is "core" als het vaak wordt geïmporteerd door andere modules
        if dependency_counts.get(module_name, 0) > 1:  # Minimaal 2 imports om als "core" te gelden
            core_modules.append(module)
        # Een module is een "workflow" als het een main-block heeft en VectorBT gebruikt
        elif module['has_main_block'] and module['uses_vectorbt']:
            workflow_modules.append(module)
        # Voeg modules die configuratie of data beheren ook toe aan core
        elif any(keyword in module['file'].lower() for keyword in ['config', 'cache', 'params']):
            core_modules.append(module)

    project_info['core_modules'] = core_modules
    project_info['workflow_modules'] = workflow_modules

    packages = defaultdict(list)
    for module in project_info['modules']:
        package_path = os.path.dirname(os.path.relpath(module['path'], project_path))
        packages[package_path].append(module)

    project_info['packages'] = [{'name': package_name or 'root', 'modules': modules} for package_name, modules in packages.items()]
    project_info['trading_pairs'] = extract_trading_pairs(python_files)
    project_info['asset_classes'] = detect_asset_classes(python_files)
    project_info['vectorbt_usage'] = extract_vectorbt_usage(python_files)
    project_info['config_files'] = extract_config_files(project_path)

    if visualize:
        graph_file = os.path.join(project_path, f"{project_info['name']}_TradingDependencies")
        project_info['dependency_graph'] = generate_dependency_graph(project_info['modules'], graph_file)

    return project_info

def generate_summary_markdown(project_info, output_file):
    """Generate a markdown summary from the project information."""
    summary = f"# {project_info['name']} - Project Summary\n\n"
    summary += f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    summary += "## Core Components\n\n"
    for module in project_info['core_modules']:
        rel_path = os.path.relpath(module['path'], project_info['path'])
        summary += f"### {module['file']} - {os.path.dirname(rel_path)}\n\n"
        if module['docstring']:
            summary += f"{module['docstring']}\n\n"

        if module['classes']:
            summary += "**Classes:**\n"
            for cls in module['classes']:
                inheritance = f" (inherits from {', '.join(cls['inheritance'])})" if cls['inheritance'] else ""
                summary += f"- **{cls['name']}{inheritance}**: {cls['docstring'].split('.')[0] if cls['docstring'] else ''}\n"
                for method in cls['methods'][:3]:
                    summary += f"    - **{method['name']}**: {method['docstring'].split('.')[0] if method['docstring'] else method['signature']}\n"
            summary += "\n"

        if module['functions']:
            summary += "**Functions:**\n"
            for func in module['functions']:
                summary += f"- **{func['name']}**: {func['docstring'].split('.')[0] if func['docstring'] else ''}\n"
            summary += "\n"

        if module['internal_depends_on']:
            summary += "**Internal Dependencies:**\n"
            for dep in module['internal_depends_on']:
                summary += f"- {dep}\n"
            summary += "\n"

    summary += "## Workflows\n\n"
    for module in project_info['workflow_modules']:
        rel_path = os.path.relpath(module['path'], project_info['path'])
        summary += f"### {module['file']} - {os.path.dirname(rel_path)}\n\n"
        if module['docstring']:
            summary += f"{module['docstring']}\n\n"
        if module['functions']:
            summary += "**Functions:**\n"
            for func in module['functions']:
                summary += f"- **{func['name']}**: {func['docstring'].split('.')[0] if func['docstring'] else ''}\n"
            summary += "\n"

    if project_info['recent_changes']:
        summary += "## Recent Changes\n"
        for change in project_info['recent_changes'][:5]:
            summary += f"- {change}\n"
        summary += "\n"

    summary += "## Asset Class Support\n"
    for asset_name, asset_info in project_info['asset_classes'].items():
        summary += f"- **{asset_name.capitalize()}**: "
        params = [f"EMA={','.join(map(str, asset_info['parameters'].get('ema_periods', [])))}",
                  f"RSI={asset_info['parameters'].get('rsi_period', '')}"]
        risks = [f"Risk={asset_info['risk_settings'].get('risk_per_trade', '')}",
                 f"Volatility Factor={asset_info['risk_settings'].get('volatility_factor', '')}" if asset_name == 'crypto' else ""]
        summary += ", ".join(filter(None, params + risks)) + "\n"
    summary += "\n"

    if project_info['trading_pairs']:
        summary += "## Supported Trading Instruments\n"
        for asset_type, pairs in project_info['trading_pairs'].items():
            if pairs:
                summary += f"- **{asset_type.capitalize()}**: {', '.join(map(str, pairs))}\n"
        summary += "\n"

    if project_info['vectorbt_usage']:
        vbt_usage = project_info['vectorbt_usage']
        summary += "## VectorBT Implementation\n"
        if vbt_usage['strategies']:
            summary += "**Strategies:**\n"
            for strat in vbt_usage['strategies']:
                summary += f"- {strat['file']}: `{strat['config']}`\n"
            summary += "\n"
        if vbt_usage['metrics']:
            summary += "**Metrics Used:**\n"
            for metric in vbt_usage['metrics']:
                summary += f"- {metric}\n"
            summary += "\n"
        if vbt_usage['portfolio_settings']:
            summary += "**Portfolio Settings:**\n"
            summary += f"- {vbt_usage['portfolio_settings']['file']}: `{vbt_usage['portfolio_settings']['settings']}`\n"
        summary += "\n"

    if project_info['config_files']:
        summary += "## Configuration Files\n"
        for config_name, config in project_info['config_files'].items():
            summary += f"- **{config_name}**: {json.dumps(config, indent=2)}\n"
        summary += "\n"

    if project_info['dependency_graph']:
        summary += "## Module Dependencies\n"
        summary += f"![Dependency Graph]({project_info['dependency_graph']})\n\n"

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(summary)
    return summary

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate a summary of a VectorBT-based trading system project')
    parser.add_argument('--path', default='.', help='Path to the project directory')
    parser.add_argument('--output', default='TradingProjectAnalysis.md', help='Output file for the summary')
    parser.add_argument('--quiet', action='store_true', help='Suppress non-critical error messages')
    parser.add_argument('--visualize', action='store_true', help='Generate a dependency graph (requires graphviz)')
    return parser.parse_args()

def main():
    """Main function for the script."""
    args = parse_args()
    global QUIET_MODE
    QUIET_MODE = args.quiet

    project_info = analyze_project(args.path, args.visualize)
    summary = generate_summary_markdown(project_info, args.output)
    print(f"Project summary written to {args.output}")
    print(f"Analyzed {len(project_info['modules'])} modules, {len(project_info['recent_changes'])} recent changes")

if __name__ == "__main__":
    main()