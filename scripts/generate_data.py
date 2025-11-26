import argparse
import os

from src.data_loader import DatasetBuilder

def main():
    parser = argparse.ArgumentParser(description="Generate Option Pricing Dataset from Market Data")
    parser.add_argument('--tickers', nargs='+', default=['^GSPC', '^NDX', '^RUT', '^DJI'],
                        help='List of tickers to include (e.g., ^GSPC ^NDX)')
    parser.add_argument('--output', type=str, default='data/processed_dataset.parquet',
                        help='Path to save the output parquet file')

    args = parser.parse_args()

    print(f"Generating dataset for tickers: {args.tickers}")
    DatasetBuilder(output_path=args.output).build()

if __name__ == "__main__":
    main()
