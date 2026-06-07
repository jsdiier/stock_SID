"""
Entry point: fetch all A-share stocks → build weekly embeddings → save to data/

Usage:
  # 存量（全量抓取，覆盖已有数据）
  python preprocess.py

  # 快速验证：只抓最近1个月
  python preprocess.py --start-date 20260506

  # 增量（只处理新数据，追加到已有文件）
  python preprocess.py --incremental

  # 只合并（跳过抓取，直接合并已有 raw/*.npz）
  python preprocess.py --merge-only
"""
import argparse
import configparser
import logging
import os
import sys

# allow imports from rqvae/ root
sys.path.insert(0, os.path.dirname(__file__))

from data_pipeline.fetcher import (
    init_api,
    get_stock_list,
    process_all_stocks,
    merge_to_dataset,
)


def _setup_logging(log_dir: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(log_dir, 'preprocess.log'), mode='a'),
        ],
    )


def main():
    parser = argparse.ArgumentParser(description='Build RQ-VAE training data')
    parser.add_argument('--config',      default='conf/common.conf')
    parser.add_argument('--incremental', action='store_true',
                        help='Append new weeks to existing per-stock files')
    parser.add_argument('--merge-only',  action='store_true',
                        help='Skip fetching; only merge existing raw/*.npz')
    parser.add_argument('--start-date',  default=None,
                        help='Override data.start_date in config (YYYYMMDD)')
    parser.add_argument('--end-date',    default=None,
                        help='Override data.end_date in config (YYYYMMDD)')
    args = parser.parse_args()

    cfg = configparser.ConfigParser()
    cfg.read(args.config, encoding='utf-8')

    _setup_logging(cfg.get('paths', 'log_dir'))
    logger = logging.getLogger(__name__)

    raw_data_dir  = cfg.get('paths', 'raw_data_dir')
    embedding_dir = cfg.get('paths', 'embedding_dir')

    if not args.merge_only:
        token      = cfg.get('data', 'tushare_token')
        start_date = args.start_date or cfg.get('data', 'start_date')
        end_date   = args.end_date   or cfg.get('data', 'end_date')
        sleep      = cfg.getfloat('data', 'api_sleep', fallback=0.3)

        pro         = init_api(token)
        stock_codes = get_stock_list(pro)
        logger.info(f"Stock list: {len(stock_codes)} stocks")

        process_all_stocks(
            pro, stock_codes, start_date, end_date,
            raw_data_dir, sleep=sleep,
            incremental=args.incremental,
        )

    merge_to_dataset(raw_data_dir, embedding_dir)


if __name__ == '__main__':
    main()
