#!/usr/bin/env python3
"""
LightRAG Runner
===============
启动脚本
"""

import argparse
import logging
from pathlib import Path

from src import LightRAGPipeline, LightRAGConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='LightRAG CLI')
    parser.add_argument('--data', type=str, default='./data', help='Data directory')
    parser.add_argument('--index', action='store_true', help='Build index')
    parser.add_argument('--query', type=str, help='Query string')
    parser.add_argument('--export', type=str, help='Export monitoring data to directory')

    args = parser.parse_args()

    # Initialize
    config = LightRAGConfig(data_path=args.data)
    rag = LightRAGPipeline(config)

    if args.index:
        logger.info("Building index...")
        stats = rag.index()
        logger.info(f"Index built: {stats}")

    if args.query:
        logger.info(f"Querying: {args.query}")
        result = rag.query(args.query)
        print(f"\nAnswer: {result['answer']}")
        print(f"\nMetrics: {result['metrics']}")

    if args.export:
        rag.export_monitoring_data(args.export)
        logger.info(f"Monitoring data exported to {args.export}")

    if not any([args.index, args.query, args.export]):
        parser.print_help()


if __name__ == "__main__":
    main()
