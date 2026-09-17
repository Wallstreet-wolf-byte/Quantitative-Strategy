"""
vnpy 参数优化脚本 - 恒指期货周期谐波震荡策略

使用方式:
    cd /workspace
    # GA遗传算法优化 (推荐, 适合大搜索空间)
    python optimize_hsi.py --csv /path/to/HSI.csv --mode ga

    # 网格搜索 (适合小范围精确搜索)
    python optimize_hsi.py --csv /path/to/HSI.csv --mode grid

    # 指定时间区间
    python optimize_hsi.py --csv /path/to/HSI.csv --mode ga --start 2024-01-01 --end 2024-06-30

    # 指定优化目标: sharpe / return / return_drawdown / win_rate
    python optimize_hsi.py --csv /path/to/HSI.csv --mode ga --target sharpe

优化完成后:
    - 控制台输出 Top 20 最优参数组合
    - 结果保存到 optimization_results.csv
"""

import sys
import time
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

from vnpy.trader.constant import Exchange, Interval, Direction, Offset
from vnpy.trader.object import BarData
from vnpy_ctastrategy.backtesting import BacktestingEngine


# ================================================================
#  配置 (与 backtest_hsi.py 保持一致)
# ================================================================

CSV_PATH = str(
    Path(__file__).resolve().parent.parent
    / "数据"
    / "HSI_2021-01-01_2026-08-29"
    / "HSI.csv"
)

SYMBOL = "HSImain"
EXCHANGE = Exchange.HKFE
INTERVAL = Interval.MINUTE

INITIAL_CAPITAL = 1_000_000
CONTRACT_SIZE = 50
PRICE_TICK = 1
COMMISSION_RATE = 0.000047
SLIPPAGE = 1

# 策略默认参数 (优化时在此基础上覆盖)
BASE_SETTING = {
    "SHORT_PERIOD": 1,
    "MIDDLE_PERIOD": 4,
    "LONG_PERIOD": 14,
    "RSV_WINDOW": 9,
    "ATR_WINDOW": 14,
    "RED_THRESHOLD": 80.0,
    "GREEN_THRESHOLD": 20.0,
    "MAX_POSITION": 1,
    "DAILY_MAX_LOSS": 1000.0,
    "POINT_VALUE": 50.0,
    "FIXED_SLIPPAGE": 1.0,
    "STOP_LOSS_MULTIPLIER": 2.0,
    "TAKE_PROFIT_MULTIPLIER": 1.5,
    "SLIPPAGE_RATIO": 0.0001,
    "LOSS_CHANCE": 2,
    "PAUSING_PERIOD": 14,
    "CLOSE_TIME_1": 180,
    "CLOSE_TIME_2": 1289,
    "DATA_WINDOW": 2000,
}

# ================================================================
#  优化参数搜索空间
# ================================================================
# 只优化影响信号质量和风控的关键参数
# 结构参数 (SHORT/MIDDLE/LONG_PERIOD) 不参与优化, 保持1/4/14

OPTIMIZATION_SPACE = {
    # KD阈值: 提高门槛减少假信号
    "RED_THRESHOLD": (75.0, 5.0, 95.0),        # start, step, end
    "GREEN_THRESHOLD": (5.0, 5.0, 25.0),

    # RSV窗口: 影响KD敏感度
    "RSV_WINDOW": (7, 1, 21),

    # ATR窗口: 影响波动率测量
    "ATR_WINDOW": (10, 2, 28),

    # 止损止盈倍数: 核心风控参数
    "STOP_LOSS_MULTIPLIER": (1.5, 0.5, 4.0),
    "TAKE_PROFIT_MULTIPLIER": (1.0, 0.5, 5.0),

    # 连亏暂停: 控制过度交易
    "LOSS_CHANCE": (2, 1, 5),
    "PAUSING_PERIOD": (7, 7, 28),               # 7/14/21/28
}

# GA优化的目标函数映射
TARGET_MAP = {
    "sharpe": "sharpe_ratio",
    "return": "total_return",
    "return_drawdown": "return_drawdown_ratio",
    "win_rate": "win_rate",
}


# ================================================================
#  数据加载 (复用 backtest_hsi.py 逻辑)
# ================================================================

def load_bars_from_csv(filepath: str, symbol: str, exchange: Exchange,
                       start: datetime, end: datetime) -> list:
    """从CSV加载K线数据"""
    print(f"正在加载CSV: {filepath}")
    df = pd.read_csv(filepath)

    col_map = {
        "时间": "datetime", "开盘价": "open", "最高价": "high",
        "最低价": "low", "收盘价": "close", "成交量": "volume",
    }
    rename = {k: v for k, v in col_map.items() if k in df.columns}
    df.rename(columns=rename, inplace=True)

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    mask = (df["datetime"] >= pd.Timestamp(start)) & \
           (df["datetime"] <= pd.Timestamp(end))
    df = df[mask].reset_index(drop=True)

    print(f"  总行数: {len(df):,}")
    if len(df) > 0:
        print(f"  时间范围: {df['datetime'].min()} → {df['datetime'].max()}")

    bars = []
    for _, row in df.iterrows():
        bar = BarData(
            symbol=symbol,
            exchange=exchange,
            datetime=row["datetime"].to_pydatetime(),
            interval=INTERVAL,
            open_price=float(row["open"]),
            high_price=float(row["high"]),
            low_price=float(row["low"]),
            close_price=float(row["close"]),
            volume=float(row.get("volume", 0)),
            turnover=0.0,
            open_interest=0.0,
        )
        bars.append(bar)

    print(f"  生成 BarData: {len(bars):,} 根")
    return bars


# ================================================================
#  优化引擎
# ================================================================

def run_optimization(
    bars: list,
    mode: str = "ga",
    target: str = "sharpe",
    start: datetime = None,
    end: datetime = None,
):
    """运行参数优化

    Args:
        bars: K线数据列表
        mode: "ga" (遗传算法) 或 "grid" (网格搜索)
        target: 优化目标
        start/end: 回测区间
    """
    from period_harmonic_strategy import PeriodHarmonicStrategy

    # 导入策略类
    engine = BacktestingEngine()
    engine.set_parameters(
        vt_symbol=f"{SYMBOL}.{EXCHANGE.value}",
        interval=INTERVAL,
        start=start,
        end=end,
        rate=COMMISSION_RATE,
        slippage=SLIPPAGE,
        size=CONTRACT_SIZE,
        pricetick=PRICE_TICK,
        capital=INITIAL_CAPITAL,
    )
    engine.add_strategy(PeriodHarmonicStrategy, BASE_SETTING)
    engine.history_data = bars
    engine.load_data = lambda: None  # 跳过数据加载, 用已注入的

    # 构建优化参数范围
    settings = {}
    for param, (s, step, e) in OPTIMIZATION_SPACE.items():
        values = []
        v = s
        while v <= e + 1e-6:
            values.append(round(v, 6))
            v += step
        settings[param] = values

    total_combos = 1
    for v in settings.values():
        total_combos *= len(v)
    print(f"\n{'='*60}")
    print(f"优化模式: {'遗传算法 (GA)' if mode == 'ga' else '网格搜索 (Grid)'}")
    print(f"优化目标: {target} ({TARGET_MAP.get(target, target)})")
    print(f"搜索空间: {total_combos:,} 种组合")
    print(f"参数范围:")
    for k, v in settings.items():
        print(f"  {k}: {v}")
    print(f"{'='*60}\n")

    t0 = time.time()

    if mode == "ga":
        # 遗传算法优化 (适合大搜索空间)
        result = engine.run_ga_optimization(
            settings=settings,
            target_name=TARGET_MAP.get(target, target),
            max_workers=4,           # 并行进程数
            population_size=50,       # 种群大小
            ngen_size=30,            # 迭代代数
            output=False,
        )
    else:
        # 网格搜索 (精确但慢)
        result = engine.run_optimization(
            settings=settings,
            target_name=TARGET_MAP.get(target, target),
            max_workers=4,
            output=False,
        )

    t1 = time.time()
    print(f"\n优化耗时: {t1 - t0:.1f} 秒")

    # 结果处理
    if result:
        df = pd.DataFrame(result)
        df = df.sort_values("value", ascending=False)

        print(f"\n{'='*60}")
        print(f"Top 20 最优参数组合 (按 {target} 排序)")
        print(f"{'='*60}")

        # 显示Top20
        for i, row in df.head(20).iterrows():
            print(f"\n--- 第 {i+1} 名 ---")
            print(f"  {target}: {row['value']:.6f}")
            for k in OPTIMIZATION_SPACE.keys():
                if k in row:
                    print(f"  {k}: {row[k]}")

        # 保存到CSV
        output_file = "optimization_results.csv"
        df.to_csv(output_file, index=False)
        print(f"\n完整结果已保存到: {output_file}")
        print(f"总结果数: {len(df)}")

        # 输出最佳参数
        best = df.iloc[0]
        print(f"\n{'='*60}")
        print(f"最佳参数组合:")
        print(f"{'='*60}")
        best_params = {}
        for k in OPTIMIZATION_SPACE.keys():
            if k in best:
                best_params[k] = best[k]
                print(f"  {k} = {best[k]}")
        print(f"  {target} = {best['value']:.6f}")

        return best_params
    else:
        print("\n优化未产生结果, 请检查参数范围和数据")
        return None


# ================================================================
#  主函数
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="vnpy 参数优化")
    parser.add_argument("--csv", type=str, default=CSV_PATH,
                        help="CSV数据文件路径")
    parser.add_argument("--mode", type=str, default="ga",
                        choices=["ga", "grid"],
                        help="优化模式: ga=遗传算法, grid=网格搜索")
    parser.add_argument("--target", type=str, default="sharpe",
                        choices=["sharpe", "return", "return_drawdown", "win_rate"],
                        help="优化目标")
    parser.add_argument("--start", type=str, default="2024-01-01",
                        help="回测开始日期 YYYY-MM-DD")
    parser.add_argument("--end", type=str, default="2024-06-30",
                        help="回测结束日期 YYYY-MM-DD")
    args = parser.parse_args()

    start_dt = datetime.strptime(args.start, "%Y-%m-%d")
    end_dt = datetime.strptime(args.end, "%Y-%m-%d")

    csv_path = args.csv

    print(f"参数优化配置:")
    print(f"  数据文件: {csv_path}")
    print(f"  优化模式: {args.mode}")
    print(f"  优化目标: {args.target}")
    print(f"  回测区间: {start_dt.date()} → {end_dt.date()}")

    # 加载数据
    bars = load_bars_from_csv(csv_path, SYMBOL, EXCHANGE, start_dt, end_dt)
    if not bars:
        print("错误: 未加载到数据, 请检查CSV路径和日期范围")
        sys.exit(1)

    # 运行优化
    best_params = run_optimization(
        bars=bars,
        mode=args.mode,
        target=args.target,
        start=start_dt,
        end=end_dt,
    )

    if best_params:
        print(f"\n建议: 用以上最佳参数运行完整回测验证")
        print(f"  python backtest_hsi.py {args.start} {args.end} --csv {csv_path}")


if __name__ == "__main__":
    main()
