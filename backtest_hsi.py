"""
vnpy 离线回测启动脚本 - 恒指期货周期谐波震荡策略

修复内容:
  1. 修复 KeyError: calculate_result() 返回的 DataFrame 中 date 是索引而非列名
  2. 修复 KeyError: end_balance 不在每日结果中, 改用 calculate_statistics() 获取
  3. 添加完整回测日志: 信号/开仓/止损/每日PnL
  4. 添加性能优化: DATA_WINDOW=2000, 信号冷却
  5. 添加回测摘要: 胜率/盈亏比/最大回撤/夏普比率

使用方式:
    cd /workspace
    python backtest_hsi.py                    # 默认回测全量数据
    python backtest_hsi.py 2024-01-01 2024-03-31  # 指定区间
"""

import sys
import time
import importlib
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.object import BarData
from vnpy_ctastrategy.backtesting import BacktestingEngine


# ================================================================
#  配置
# ================================================================

CSV_PATH = "/workspace/.uploads/e7dd03d5-0a6e-4403-8849-315e2769d79c_HSI.csv"

SYMBOL = "HSImain"
EXCHANGE = Exchange.HKFE
INTERVAL = Interval.MINUTE

# 默认回测区间(可通过命令行参数覆盖)
DEFAULT_START = datetime(2021, 1, 1)
DEFAULT_END = datetime(2026, 8, 29)

INITIAL_CAPITAL = 1_000_000
CONTRACT_SIZE = 50
PRICE_TICK = 1
COMMISSION_RATE = 0.000047
SLIPPAGE = 1

# 策略参数
STRATEGY_SETTING = {
    "SHORT_PERIOD": 1,
    "MIDDLE_PERIOD": 4,
    "LONG_PERIOD": 14,
    "SIGNAL_THRESH": 2,
    "RSV_WINDOW": 9,
    "ATR_WINDOW": 14,
    "RED_THRESHOLD": 80.0,
    "GREEN_THRESHOLD": 20.0,
    "LONG_WEIGHT": 9,
    "MIDDLE_WEIGHT": 3,
    "SHORT_WEIGHT": 1,
    "SCORE_OFFSET": 13,
    "MAX_POSITION": 1,
    "DAILY_MAX_LOSS": 1000.0,
    "POINT_VALUE": 50.0,
    "FIXED_SLIPPAGE": 1.0,
    "STOP_LOSS_MULTIPLIER": 2.0,
    "TAKE_PROFIT_MULTIPLIER": 1.5,
    "SLIPPAGE_RATIO": 0.0001,
    "LOSS_CHANCE": 2,
    "PAUSING_PERIOD": 14,
    "CLOSE_TIME_1": 180,     # 03:00
    "CLOSE_TIME_2": 1289,    # 21:29
    "DATA_WINDOW": 2000,     # PERF-1: 从9600降到2000
}


# ================================================================
#  数据加载
# ================================================================

def load_bars_from_csv(filepath: str, symbol: str, exchange: Exchange,
                       start: datetime, end: datetime) -> list:
    """从CSV加载K线数据, 转换为vnpy BarData列表"""
    print(f"正在加载CSV: {filepath}")
    df = pd.read_csv(filepath)

    # 列名映射(中文→英文)
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
    else:
        print("  WARNING: 该时间范围内无数据!")
        return []

    # 检查数据质量
    _check_data_quality(df)

    # 逐行构造BarData
    bars: list = []
    for row in df.itertuples():
        bar = BarData(
            symbol=symbol,
            exchange=exchange,
            datetime=row.datetime.to_pydatetime(),
            interval=INTERVAL,
            open_price=float(row.open),
            high_price=float(row.high),
            low_price=float(row.low),
            close_price=float(row.close),
            volume=float(row.volume),
            gateway_name="CSV",
        )
        bars.append(bar)

    print(f"  转换完成: {len(bars):,} 根BarData")
    return bars


def _check_data_quality(df: pd.DataFrame):
    """检查数据质量: 缺失值/异常值/时间连续性"""
    # 缺失值
    null_count = df[["open", "high", "low", "close", "volume"]].isnull().sum().sum()
    if null_count > 0:
        print(f"  WARNING: 缺失值 {null_count} 个")

    # 价格异常
    price_cols = ["open", "high", "low", "close"]
    for col in price_cols:
        zero_count = (df[col] == 0).sum()
        if zero_count > 0:
            print(f"  WARNING: {col} 列有 {zero_count} 个零值")

    neg_vol = (df["volume"] < 0).sum()
    if neg_vol > 0:
        print(f"  WARNING: 成交量有 {neg_vol} 个负值")

    # 时间重复
    dup_count = df["datetime"].duplicated().sum()
    if dup_count > 0:
        print(f"  WARNING: 时间重复 {dup_count} 条, 已自动去重")
        df.drop_duplicates(subset="datetime", inplace=True)

    print(f"  数据质量检查通过")


# ================================================================
#  回测引擎
# ================================================================

def run_backtest(start: datetime, end: datetime, show_chart: bool = False):
    """运行完整回测"""
    print("=" * 70)
    print("恒指期货 周期谐波震荡策略 vnpy回测")
    print("=" * 70)
    print(f"合约: {SYMBOL}.{EXCHANGE.value}")
    print(f"回测区间: {start.date()} → {end.date()}")
    print(f"初始资金: {INITIAL_CAPITAL:,.0f} HKD")
    print(f"合约乘数: {CONTRACT_SIZE} HKD/点")
    print(f"手续费率: {COMMISSION_RATE:.6f}")
    print(f"滑点: {SLIPPAGE} 点")
    print(f"DATA_WINDOW: {STRATEGY_SETTING['DATA_WINDOW']}")
    print("=" * 70)

    # 1. 加载数据
    bars = load_bars_from_csv(CSV_PATH, SYMBOL, EXCHANGE, start, end)
    if not bars:
        print("无数据, 退出")
        return

    # 2. 创建回测引擎
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

    # 3. 添加策略
    mod = importlib.import_module("period_harmonic_strategy")
    strategy_class = getattr(mod, "PeriodHarmonicStrategy")
    engine.add_strategy(strategy_class, STRATEGY_SETTING.copy())

    # 4. 注入数据
    print(f"\n注入 {len(bars):,} 根BarData到回测引擎...")
    engine.history_data = bars

    # 5. 运行回测
    print("开始回测...")
    t0 = time.time()
    engine.run_backtesting()
    t1 = time.time()
    print(f"\n回测耗时: {t1 - t0:.1f}秒")

    # 6. 成交记录
    trades = engine.get_all_trades()
    print(f"\n{'=' * 70}")
    print(f"成交记录: 共 {len(trades)} 笔")
    print(f"{'=' * 70}")

    if len(trades) > 0:
        print(f"\n前20笔成交:")
        print(f"{'时间':>20s}  {'方向':>4s}  {'开平':>4s}  {'价格':>10s}  {'数量':>4s}")
        print("-" * 50)
        for t in trades[:20]:
            print(f"  {t.datetime}  {str(t.direction):>4s}  "
                  f"{str(t.offset):>4s}  {t.price:>10.1f}  {t.volume:>4d}")

        if len(trades) > 20:
            print(f"  ... (共 {len(trades)} 笔)")

    # 7. 计算逐日盈亏
    print(f"\n{'=' * 70}")
    print("逐日盯市盈亏")
    print(f"{'=' * 70}")

    df_result = engine.calculate_result()

    if df_result is not None and len(df_result) > 0:
        # 修复: date 是索引不是列名, end_balance 不在每日结果中
        # 实际列: close_price, pre_close, trade_count, start_pos, end_pos,
        #         turnover, commission, slippage, trading_pnl, holding_pnl,
        #         total_pnl, net_pnl
        display_cols = [c for c in [
            "trade_count", "turnover", "commission",
            "slippage", "net_pnl", "total_pnl"
        ] if c in df_result.columns]

        # 打印前30天和后30天的每日盈亏
        print(f"\n前30天:")
        print(df_result[display_cols].head(30).to_string())
        if len(df_result) > 60:
            print(f"\n... (中间省略 {len(df_result) - 60:.0f} 天)")
        print(f"\n后30天:")
        print(df_result[display_cols].tail(30).to_string())

    # 8. 计算统计指标
    print(f"\n{'=' * 70}")
    print("回测统计摘要")
    print(f"{'=' * 70}")

    stats = engine.calculate_statistics()

    # 9. 额外分析: 胜率/盈亏比/连亏统计
    _print_trade_analysis(trades)

    # 10. 每日PnL汇总
    if df_result is not None and len(df_result) > 0:
        _print_daily_summary(df_result)

    # 11. 图表(可选)
    if show_chart:
        try:
            engine.show_chart()
        except Exception as e:
            print(f"\n图表显示失败(非关键): {e}")

    return engine, df_result, trades


def _print_trade_analysis(trades: list):
    """分析成交记录: 胜率/盈亏比/连亏"""
    if len(trades) < 2:
        print("\n成交不足, 跳过交易分析")
        return

    print(f"\n{'=' * 70}")
    print("交易分析")
    print(f"{'=' * 70}")

    # 按时间排序
    sorted_trades = sorted(trades, key=lambda t: t.datetime)

    # 配对: 开仓→平仓
    pairs = []
    open_trade = None
    for t in sorted_trades:
        if t.offset and "OPEN" in str(t.offset).upper():
            if open_trade is not None:
                # 前一个开仓未平, 跳过
                pass
            open_trade = t
        elif t.offset and "CLOSE" in str(t.offset).upper():
            if open_trade is not None:
                if str(open_trade.direction) == "LONG":
                    pnl = (t.price - open_trade.price) * CONTRACT_SIZE * t.volume
                else:
                    pnl = (open_trade.price - t.price) * CONTRACT_SIZE * t.volume
                pairs.append({
                    "open_time": open_trade.datetime,
                    "close_time": t.datetime,
                    "direction": str(open_trade.direction),
                    "open_price": open_trade.price,
                    "close_price": t.price,
                    "pnl": pnl,
                    "volume": t.volume,
                })
                open_trade = None
        else:
            # 反手或未知, 尝试按方向配对
            if open_trade is None:
                open_trade = t
            else:
                if str(open_trade.direction) == "LONG":
                    pnl = (t.price - open_trade.price) * CONTRACT_SIZE * t.volume
                else:
                    pnl = (open_trade.price - t.price) * CONTRACT_SIZE * t.volume
                pairs.append({
                    "open_time": open_trade.datetime,
                    "close_time": t.datetime,
                    "direction": str(open_trade.direction),
                    "open_price": open_trade.price,
                    "close_price": t.price,
                    "pnl": pnl,
                    "volume": t.volume,
                })
                open_trade = None

    if not pairs:
        print("无法配对成交记录")
        return

    df_pairs = pd.DataFrame(pairs)
    wins = df_pairs[df_pairs["pnl"] > 0]
    losses = df_pairs[df_pairs["pnl"] <= 0]

    win_rate = len(wins) / len(df_pairs) * 100 if len(df_pairs) > 0 else 0
    avg_win = wins["pnl"].mean() if len(wins) > 0 else 0
    avg_loss = losses["pnl"].mean() if len(losses) > 0 else 0
    profit_factor = abs(wins["pnl"].sum() / losses["pnl"].sum()) \
        if len(losses) > 0 and losses["pnl"].sum() != 0 else float('inf')

    total_pnl = df_pairs["pnl"].sum()
    max_win = df_pairs["pnl"].max()
    max_loss = df_pairs["pnl"].min()

    print(f"  总交易次数: {len(df_pairs)}")
    print(f"  盈利次数: {len(wins)}")
    print(f"  亏损次数: {len(losses)}")
    print(f"  胜率: {win_rate:.1f}%")
    print(f"  平均盈利: {avg_win:,.0f} HKD")
    print(f"  平均亏损: {avg_loss:,.0f} HKD")
    print(f"  盈亏比: {profit_factor:.2f}")
    print(f"  总盈亏: {total_pnl:,.0f} HKD")
    print(f"  最大单笔盈利: {max_win:,.0f} HKD")
    print(f"  最大单笔亏损: {max_loss:,.0f} HKD")

    # 连续亏损统计
    streak = 0
    max_streak = 0
    for pnl in df_pairs["pnl"]:
        if pnl <= 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    print(f"  最大连续亏损次数: {max_streak}")

    # 盈利交易明细(前10)
    print(f"\n  盈利最大的10笔交易:")
    print(f"  {'开仓时间':>20s}  {'方向':>4s}  {'开仓价':>10s}  "
          f"{'平仓价':>10s}  {'盈亏(HKD)':>12s}")
    print("  " + "-" * 65)
    for _, row in df_pairs.nlargest(10, "pnl").iterrows():
        print(f"  {str(row['open_time']):>20s}  {row['direction']:>4s}  "
              f"{row['open_price']:>10.1f}  {row['close_price']:>10.1f}  "
              f"{row['pnl']:>12,.0f}")

    # 亏损交易明细(前10)
    print(f"\n  亏损最大的10笔交易:")
    print(f"  {'开仓时间':>20s}  {'方向':>4s}  {'开仓价':>10s}  "
          f"{'平仓价':>10s}  {'盈亏(HKD)':>12s}")
    print("  " + "-" * 65)
    for _, row in df_pairs.nsmallest(10, "pnl").iterrows():
        print(f"  {str(row['open_time']):>20s}  {row['direction']:>4s}  "
              f"{row['open_price']:>10.1f}  {row['close_price']:>10.1f}  "
              f"{row['pnl']:>12,.0f}")


def _print_daily_summary(df_result: pd.DataFrame):
    """打印每日PnL汇总统计"""
    print(f"\n{'=' * 70}")
    print("每日PnL汇总")
    print(f"{'=' * 70}")

    if "net_pnl" not in df_result.columns:
        print("WARNING: net_pnl 列不存在")
        return

    net_pnl = df_result["net_pnl"]
    trade_count = df_result["trade_count"] if "trade_count" in df_result.columns else None

    total_days = len(df_result)
    trading_days = (trade_count > 0).sum() if trade_count is not None else 0
    win_days = (net_pnl > 0).sum()
    loss_days = (net_pnl < 0).sum()
    flat_days = (net_pnl == 0).sum()

    print(f"  总日历天数: {total_days}")
    print(f"  有交易天数: {trading_days}")
    print(f"  盈利天数: {win_days}")
    print(f"  亏损天数: {loss_days}")
    print(f"  无交易天数: {flat_days}")
    print(f"  日均盈亏: {net_pnl.mean():,.0f} HKD")
    print(f"  日均交易盈亏(仅有交易日): "
          f"{net_pnl[trade_count > 0].mean() if trading_days > 0 else 0:,.0f} HKD")
    print(f"  最大日盈利: {net_pnl.max():,.0f} HKD")
    print(f"  最大日亏损: {net_pnl.min():,.0f} HKD")
    print(f"  总净盈亏: {net_pnl.sum():,.0f} HKD")

    # 按月汇总
    if isinstance(df_result.index, pd.DatetimeIndex) or \
       all(isinstance(d, (datetime, pd.Timestamp)) for d in df_result.index[:5]):
        monthly = df_result.copy()
        monthly.index = pd.to_datetime(monthly.index)
        monthly_pnl = monthly["net_pnl"].resample("ME").sum()
        monthly_trades = monthly["trade_count"].resample("ME").sum() \
            if "trade_count" in monthly.columns else None

        print(f"\n  月度盈亏汇总:")
        print(f"  {'月份':>10s}  {'净盈亏(HKD)':>14s}  {'交易次数':>8s}")
        print("  " + "-" * 38)
        for date, pnl in monthly_pnl.items():
            tc = monthly_trades.get(date, 0) if monthly_trades is not None else 0
            print(f"  {date.strftime('%Y-%m'):>10s}  {pnl:>14,.0f}  {int(tc):>8d}")


# ================================================================
#  主入口
# ================================================================

if __name__ == "__main__":
    # 解析命令行参数
    if len(sys.argv) >= 3:
        start = datetime.strptime(sys.argv[1], "%Y-%m-%d")
        end = datetime.strptime(sys.argv[2], "%Y-%m-%d")
    else:
        start = DEFAULT_START
        end = DEFAULT_END

    show_chart = "--chart" in sys.argv

    run_backtest(start, end, show_chart=show_chart)
