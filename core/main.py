# core/main.py
import argparse
import sys
from pathlib import Path
from typing import List

# 确保项目根目录在Python路径中
sys.path.append(str(Path(__file__).parent.parent))

from core.utils.log import logger
from core.stock.tushare_data_provider import get_technical_factor  # 假设已实现数据接口
from stock_predictor import run_stock_prediction, run_single_stock_prediction  # 从catboost.py导入核心函数


def configure_logging():
    """配置日志系统（已在core.utils.log中实现则无需重复）"""
    logger.info("Logger initialized")


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="基于CatBoost的股票价格预测系统",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--symbols",
                        nargs="+",
                        default=["601127"],
                        help="股票代码列表，例如：601127 600519")
    parser.add_argument("--days", type=int, default=400, help="使用的历史数据天数")
    parser.add_argument("--optimize",
                        action="store_true",
                        help="启用Optuna超参数优化")
    parser.add_argument("--max_workers",
                        type=int,
                        default=2,
                        help="并行任务数（根据GPU数量调整）")
    parser.add_argument("--save_models",
                        action="store_true",
                        help="保存训练好的模型到本地")
    parser.add_argument("--single", action="store_true", help="单股票模式（强制关闭并行）")
    return parser.parse_args()


def main():
    args = parse_arguments()
    configure_logging()

    logger.info(f"开始预测任务，参数配置: {vars(args)}")

    try:
        if args.single or len(args.symbols) == 1:
            # 单股票模式
            symbol = args.symbols[0]
            _, result = run_single_stock_prediction(
                symbol=symbol,
                days=args.days,
                optimize=args.optimize,
                save_model=args.save_models)
            logger.info(f"预测完成: {symbol} | 下期价格: {result['next_price']:.2f}")
        else:
            # 多股票并行模式
            results = run_stock_prediction(symbols=args.symbols,
                                           days=args.days,
                                           optimize=args.optimize,
                                           max_workers=args.max_workers,
                                           save_models=args.save_models)
            logger.info(
                f"批量预测完成，总计成功: {sum(1 for r in results.values() if 'error' not in r)}"
            )

    except Exception as e:
        logger.error(f"主程序运行异常: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
