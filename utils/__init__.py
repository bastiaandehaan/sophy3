# path from content root [utils/__init__.py]
"""
Sophy3 - Utilities
Functie: Algemene hulpfuncties voor het framework
Auteur: AI Trading Assistant
Laatste update: 2025-04-02

Gebruik:
  Algemene hulpfuncties en tools voor het framework,
  inclusief cache management functionaliteit.

Dependencies:
  - argparse
  - os
"""

import argparse
import os
import logging

logger = logging.getLogger(__name__)


def show_cache_stats():
    """Toont statistieken over de gecachte data."""
    # Import hier om circulaire imports te voorkomen
    from data.cache import CACHE_DIR

    total_size = 0
    file_count = 0
    asset_stats = {}

    # Check of cache directory bestaat
    if not os.path.exists(CACHE_DIR):
        print(f"Cache directory bestaat niet: {CACHE_DIR}")
        return 0, 0

    # Loop door alle bestanden in de cache directory
    for root, dirs, files in os.walk(CACHE_DIR):
        for file in files:
            if file.endswith('.parquet'):
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                total_size += file_size
                file_count += 1

                # Bepaal asset class uit pad
                path_parts = os.path.relpath(root, CACHE_DIR).split(os.sep)
                if len(path_parts) > 0:
                    asset_class = path_parts[0]
                    asset_stats[asset_class] = asset_stats.get(asset_class, 0) + 1

    # Print statistieken
    print("\n=== Cache Statistieken ===")
    print(f"Cache locatie: {CACHE_DIR}")
    print(f"Totaal aantal bestanden: {file_count}")
    print(f"Totale grootte: {total_size / (1024 * 1024):.2f} MB")

    print("\nBestanden per asset class:")
    for asset, count in asset_stats.items():
        print(f"  {asset}: {count} bestanden")

    return file_count, total_size


def cache_management_cli():
    """Command-line interface voor cache management."""
    # Import hier om circulaire imports te voorkomen
    from data.cache import clear_cache

    parser = argparse.ArgumentParser(description='Sophy3 Cache Management Tool')
    parser.add_argument('--clear', action='store_true',
                        help='Verwijder cache bestanden')
    parser.add_argument('--symbol', type=str,
                        help='Specifiek symbool om te verwijderen')
    parser.add_argument('--timeframe', type=str, help='Specifieke timeframe (bv. H1)')
    parser.add_argument('--asset-class', type=str,
                        choices=['forex', 'crypto', 'stocks', 'indices'],
                        help='Specifieke asset class')
    parser.add_argument('--older-than', type=int,
                        help='Verwijder alleen bestanden ouder dan X dagen')
    parser.add_argument('--stats', action='store_true', help='Toon cache statistieken')

    args = parser.parse_args()

    if args.stats:
        show_cache_stats()

    if args.clear:
        count = clear_cache(args.symbol, args.timeframe, args.asset_class,
                            args.older_than)
        print(f"Cache opgeschoond: {count} bestanden verwijderd.")

        if args.stats:
            print("\nNieuwe cache statistieken:")
            show_cache_stats()


if __name__ == "__main__":
    cache_management_cli()
