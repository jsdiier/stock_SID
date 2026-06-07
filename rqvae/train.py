"""
Entry point: train RQ-VAE on preprocessed weekly embeddings.

Usage:
  # 存量训练（从头开始）
  python train.py

  # 从最新 checkpoint 恢复
  python train.py --resume

  # 增量训练（加载已有模型，继续训练新数据）
  python train.py --mode incremental --resume

  # 指定配置文件
  python train.py --config conf/common.conf
"""
import argparse
import configparser
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from trainer.trainer import Trainer


def _setup_logging(log_dir: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(log_dir, 'train.log'), mode='a'),
        ],
    )


def main():
    parser = argparse.ArgumentParser(description='Train RQ-VAE')
    parser.add_argument('--config', default='conf/common.conf')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from latest checkpoint')
    parser.add_argument('--mode',   choices=['full', 'incremental'],
                        help='Override training.mode in config')
    args = parser.parse_args()

    cfg = configparser.ConfigParser()
    cfg.read(args.config, encoding='utf-8')

    if args.mode:
        cfg.set('training', 'mode', args.mode)

    _setup_logging(cfg.get('paths', 'log_dir'))
    logger = logging.getLogger(__name__)

    mode = cfg.get('training', 'mode')
    logger.info(f"Training mode: {mode}")

    trainer = Trainer(cfg)
    trainer.train(resume=args.resume or mode == 'incremental')


if __name__ == '__main__':
    main()
