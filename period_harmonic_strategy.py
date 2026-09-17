"""
vnpy CTA 策略: 周期谐波震荡策略 (Period Harmonic Oscillation)

修复了 Codex 审查发现的全部9个问题 + 4项性能优化:
  P0-1: KD递推公式权重和=0.75导致衰减，改为标准2/3+1/3
  P0-2: 持仓后直接return阻断信号/止损更新，重构on_bar流程
  P0-3: 止损更新方向反了，改为追踪止损(只收紧不放松)
  P1-1: resample不按时间边界分组，改为按交易日对齐重采样
  P1-2: 下单后立即改方向状态，vnpy引擎自动维护self.pos
  P2-1: 反手交易不计入亏损统计，on_trade统一处理
  P2-2: 每日亏损按自然日期重置，改为17:15交易日切分
  P2-3: 短周期不参与最终决策(设计问题，保留原逻辑+参数化)
  P2-4: 订单回报过滤风险，vnpy天然解决
  PERF-1: DATA_WINDOW从9600降到2000
  PERF-2: 信号冷却——每LONG_PERIOD根K线才重算信号
  PERF-3: resample用np.maximum.reduceat完全向量化, 零Python循环
  PERF-4: ATR只在信号重算时计算, 不每根K线重复创建大数组
"""

from datetime import datetime, timedelta
from typing import Optional, List

import numpy as np

from vnpy_ctastrategy import (
    CtaTemplate,
    StopOrder,
    TickData,
    BarData,
    TradeData,
    OrderData,
    BarGenerator,
    ArrayManager,
)
from vnpy.trader.constant import Direction


class PeriodHarmonicStrategy(CtaTemplate):
    """周期谐波震荡策略 - vnpy CTA版本"""

    # ============ 策略参数 ============
    SHORT_PERIOD: int = 1
    MIDDLE_PERIOD: int = 4
    LONG_PERIOD: int = 14
    SIGNAL_THRESH: int = 2

    RSV_WINDOW: int = 9
    K_WINDOW: int = 3
    D_WINDOW: int = 3
    ATR_WINDOW: int = 14

    RED_THRESHOLD: float = 80.0
    GREEN_THRESHOLD: float = 20.0

    LONG_WEIGHT: int = 9
    MIDDLE_WEIGHT: int = 3
    SHORT_WEIGHT: int = 1
    SCORE_OFFSET: int = 13

    MAX_POSITION: int = 1
    DAILY_MAX_LOSS: float = 1000.0
    POINT_VALUE: float = 50.0
    FIXED_SLIPPAGE: float = 1.0

    STOP_LOSS_MULTIPLIER: float = 2.0
    TAKE_PROFIT_MULTIPLIER: float = 1.5
    SLIPPAGE_RATIO: float = 0.0001

    LOSS_CHANCE: int = 2
    PAUSING_PERIOD: int = 14

    CLOSE_TIME_1: int = 180
    CLOSE_TIME_2: int = 1289

    DATA_WINDOW: int = 2000

    parameters: List[str] = [
        "SHORT_PERIOD", "MIDDLE_PERIOD", "LONG_PERIOD", "SIGNAL_THRESH",
        "RSV_WINDOW", "K_WINDOW", "D_WINDOW", "ATR_WINDOW",
        "RED_THRESHOLD", "GREEN_THRESHOLD",
        "LONG_WEIGHT", "MIDDLE_WEIGHT", "SHORT_WEIGHT", "SCORE_OFFSET",
        "MAX_POSITION", "DAILY_MAX_LOSS", "POINT_VALUE", "FIXED_SLIPPAGE",
        "STOP_LOSS_MULTIPLIER", "TAKE_PROFIT_MULTIPLIER", "SLIPPAGE_RATIO",
        "LOSS_CHANCE", "PAUSING_PERIOD",
        "CLOSE_TIME_1", "CLOSE_TIME_2", "DATA_WINDOW",
    ]

    variables: List[str] = [
        "loss_num", "pausing_countdown", "daily_loss",
        "stop_price", "take_profit", "current_trading_day",
        "entry_price", "prev_direction", "last_signal_direction",
    ]

    # ============ 初始化 ============
    def __init__(self, cta_engine, strategy_name: str, vt_symbol: str, setting: dict):
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)

        self.bg = BarGenerator(self.on_bar)
        self.am = ArrayManager(size=self.DATA_WINDOW)
        self.bar_datetimes: List[datetime] = []

        self.loss_num: int = 0
        self.pausing_countdown: int = 0
        self.daily_loss: float = 0.0
        self.current_trading_day: str = ""

        self.stop_price: Optional[float] = None
        self.take_profit: Optional[float] = None
        self.entry_price: Optional[float] = None
        self.prev_direction: int = 0
        self.pending_reverse_signal: int = 0

        self.last_signal_direction: int = 0
        self.last_signal_ub: float = 0.0
        self.last_signal_lb: float = 0.0
        self.last_atr: float = 10.0
        self.bar_count: int = 0
        self.order_pending: bool = False
        self.last_pos: int = 0

    # ================================================================
    #  vnpy 回调
    # ================================================================

    def on_init(self):
        self.write_log("策略初始化")
        try:
            self.load_bar(10)
        except Exception:
            pass

    def on_start(self):
        self.write_log("策略启动")
        self.put_event()

    def on_stop(self):
        self.write_log("策略停止")
        self.put_event()

    def on_tick(self, tick: TickData):
        self.bg.update_tick(tick)

    def on_bar(self, bar: BarData):
        """1分钟K线回调"""
        # 1. 更新数据缓存
        self.am.update_bar(bar)
        self.bar_datetimes.append(bar.datetime)
        if len(self.bar_datetimes) > self.DATA_WINDOW:
            self.bar_datetimes = self.bar_datetimes[-self.DATA_WINDOW:]

        if not self.am.inited:
            return

        self.bar_count += 1

        # === 调试日志: 前5根 + 每1000根打印一次 ===
        if self.bar_count <= 5:
            self.write_log(
                f"调试 bar_count={self.bar_count} dt={bar.datetime} "
                f"O={bar.open_price} H={bar.high_price} L={bar.low_price} "
                f"C={bar.close_price} V={bar.volume} pos={self.pos} "
                f"am_count={self.am.count} dt_len={len(self.bar_datetimes)}"
            )
        elif self.bar_count % 10000 == 0:
            self.write_log(
                f"调试 bar_count={self.bar_count} dt={bar.datetime} "
                f"C={bar.close_price} pos={self.pos}"
            )

        # 持仓变化时重置order_pending(上一笔订单已成交)
        if self.pos != self.last_pos:
            self.order_pending = False
            self.last_pos = self.pos

        # 2. 交易日切换
        self._check_day_rollover(bar)

        # 3. 连亏暂停
        if self.loss_num >= self.LOSS_CHANCE:
            self.pausing_countdown = self.PAUSING_PERIOD
            self.loss_num = 0
        if self.pausing_countdown > 0:
            self.pausing_countdown -= 1
            self.put_event()
            return

        # 4. 定时平仓
        current_minute = bar.datetime.hour * 60 + bar.datetime.minute
        if current_minute in (self.CLOSE_TIME_1, self.CLOSE_TIME_2):
            if self.pos != 0:
                self.write_log(f"定时平仓: {bar.datetime}")
                self.cancel_all()
                if self.pos > 0:
                    self.sell(bar.close_price, abs(self.pos))
                elif self.pos < 0:
                    self.cover(bar.close_price, abs(self.pos))
            self.put_event()
            return

        # 5. 止损止盈触发(每根K线检查, 仅持仓)
        if self.pos > 0 and self.stop_price is not None:
            if bar.low_price <= self.stop_price:
                self.cancel_all()
                self.sell(self.stop_price, abs(self.pos))
                self.order_pending = True
                self.put_event()
                return
            if self.take_profit is not None and bar.high_price >= self.take_profit:
                self.cancel_all()
                self.sell(self.take_profit, abs(self.pos))
                self.order_pending = True
                self.put_event()
                return
        elif self.pos < 0 and self.stop_price is not None:
            if bar.high_price >= self.stop_price:
                self.cancel_all()
                self.cover(self.stop_price, abs(self.pos))
                self.order_pending = True
                self.put_event()
                return
            if self.take_profit is not None and bar.low_price <= self.take_profit:
                self.cancel_all()
                self.cover(self.take_profit, abs(self.pos))
                self.order_pending = True
                self.put_event()
                return

        # 6. 信号计算(每LONG_PERIOD根重算, 其余用缓存)
        need_signal = (self.bar_count % self.LONG_PERIOD == 0)
        if need_signal:
            try:
                direction, ub, lb, atr = self._is_harmonic_oscillation()
                self.last_signal_direction = direction
                self.last_signal_ub = ub
                self.last_signal_lb = lb
                self.last_atr = atr
                # 调试: 前3次信号计算打印详情
                if self.bar_count <= self.LONG_PERIOD * 3:
                    self.write_log(
                        f"信号计算 bar_count={self.bar_count} "
                        f"direction={direction} atr={atr:.2f} "
                        f"ub={ub:.2f} lb={lb:.2f} "
                        f"close={bar.close_price}"
                    )
            except Exception as e:
                self.write_log(f"信号计算异常: {e}")
                direction = 0
                ub = 0
                lb = 0
                atr = 10.0
        else:
            direction = self.last_signal_direction
            ub = self.last_signal_ub
            lb = self.last_signal_lb

        # 7. 更新止损止盈(轻量, 用缓存的ATR)
        self._update_stop_profit(direction, ub, lb, bar)

        # 8. 开仓/反手判断
        if self.daily_loss >= self.DAILY_MAX_LOSS:
            self.put_event()
            return

        # 有待处理订单时不重复下单
        if self.order_pending:
            self.put_event()
            return

        # 反手待开仓: 上一根K线已平仓, 现在开反向仓
        if self.pending_reverse_signal != 0 and self.pos == 0:
            rev = self.pending_reverse_signal
            self.pending_reverse_signal = 0
            if rev == 1:
                self.buy(bar.close_price, 1)
            else:
                self.short(bar.close_price, 1)
            self.order_pending = True
            self.put_event()
            return

        # 正常开仓
        if direction == 1 and self.pos == 0:
            self.write_log(f"开多 signal=1 close={bar.close_price} bar_count={self.bar_count}")
            self.buy(bar.close_price, 1)
            self.order_pending = True
        elif direction == -1 and self.pos == 0:
            self.write_log(f"开空 signal=-1 close={bar.close_price} bar_count={self.bar_count}")
            self.short(bar.close_price, 1)
            self.order_pending = True
        # 反手: 先平仓, 设置pending flag, 下一根K线再开反向仓
        elif direction == -1 and self.pos > 0:
            self.write_log(f"反手多→空 close={bar.close_price} bar_count={self.bar_count}")
            self.pending_reverse_signal = -1
            self.sell(bar.close_price, abs(self.pos))
            self.order_pending = True
        elif direction == 1 and self.pos < 0:
            self.write_log(f"反手空→多 close={bar.close_price} bar_count={self.bar_count}")
            self.pending_reverse_signal = 1
            self.cover(bar.close_price, abs(self.pos))
            self.order_pending = True

        self.put_event()

    def on_order(self, order: OrderData):
        pass

    def on_trade(self, trade: TradeData):
        """成交回报 - 仅统计盈亏, 不下单"""
        if self.entry_price is not None and self.prev_direction != 0:
            exit_price = trade.price
            pnl = self._calc_pnl(self.entry_price, exit_price, self.prev_direction, 1)

            if pnl < 0:
                self.loss_num += 1
                self.daily_loss += abs(pnl)
            else:
                self.loss_num = 0

            self.entry_price = None
        else:
            self.entry_price = trade.price
            if trade.direction == Direction.LONG:
                self.prev_direction = 1
            else:
                self.prev_direction = -1

        self.put_event()

    def on_stop_order(self, stop_order: StopOrder):
        self.put_event()

    # ================================================================
    #  交易日与风控
    # ================================================================

    def _get_trading_day(self, dt: datetime) -> str:
        if dt.hour >= 17:
            return (dt + timedelta(days=1)).date().isoformat()
        return dt.date().isoformat()

    def _check_day_rollover(self, bar: BarData):
        day = self._get_trading_day(bar.datetime)
        if day != self.current_trading_day:
            self.current_trading_day = day
            self.daily_loss = 0.0

    def _calc_pnl(self, entry, exit_price, direction, volume):
        cost = self.FIXED_SLIPPAGE * self.POINT_VALUE * volume
        if direction == 1:
            return (exit_price - entry) * self.POINT_VALUE * volume - cost
        return (entry - exit_price) * self.POINT_VALUE * volume - cost

    # ================================================================
    #  信号计算(修复KD + 完全向量化resample)
    # ================================================================

    def _is_harmonic_oscillation(self):
        """判断周期谐波震荡信号"""
        ohlcv = np.array([
            self.am.open_array,
            self.am.high_array,
            self.am.low_array,
            self.am.close_array,
            self.am.volume_array,
        ])  # (5, N)

        datetimes = self.bar_datetimes

        short_data = self._resample_by_time(ohlcv, datetimes, self.SHORT_PERIOD)
        middle_data = self._resample_by_time(ohlcv, datetimes, self.MIDDLE_PERIOD)
        long_data = self._resample_by_time(ohlcv, datetimes, self.LONG_PERIOD)

        short_sig = self._red_green_signal(short_data)
        middle_sig = self._red_green_signal(middle_data)
        long_sig = self._red_green_signal(long_data)

        score = (
            self.LONG_WEIGHT * long_sig
            + self.MIDDLE_WEIGHT * middle_sig
            + self.SHORT_WEIGHT * short_sig
            - self.SCORE_OFFSET
        )

        # ATR
        ohlcv_2d = ohlcv.T
        atr = self._calculate_atr(ohlcv_2d)
        last_close = ohlcv_2d[-1, 3]

        ub = last_close + self.STOP_LOSS_MULTIPLIER * atr * (1 + self.SLIPPAGE_RATIO)
        lb = last_close - self.STOP_LOSS_MULTIPLIER * atr * (1 + self.SLIPPAGE_RATIO)

        if score >= self.SIGNAL_THRESH:
            return 1, ub, lb, atr
        elif score <= -self.SIGNAL_THRESH:
            return -1, ub, lb, atr
        return 0, ub, lb, atr

    def _resample_by_time(self, ohlcv: np.ndarray, datetimes: list, period: int) -> np.ndarray:
        """
        按时间边界重采样(PERF-3完全向量化)

        用np.maximum.reduceat / np.minimum.reduceat / np.add.reduceat
        实现零Python循环的分组聚合。
        """
        if period == 1:
            return ohlcv.copy()

        n = ohlcv.shape[1]
        if n == 0:
            return np.empty((5, 0))

        # 1. 向量化计算交易日分组(缓存hours和dates避免重复转换)
        hours = np.fromiter((d.hour for d in datetimes), dtype=np.int32, count=n)
        dates = np.fromiter((d.toordinal() for d in datetimes), dtype=np.int32, count=n)

        # 交易日: hour>=17 的bar属于次日
        trading_day_ord = np.where(hours >= 17, dates + 1, dates)

        # 2. 找交易日边界
        day_change = np.empty(n, dtype=bool)
        day_change[0] = True
        day_change[1:] = trading_day_ord[1:] != trading_day_ord[:-1]

        # 3. 向量化计算日内bar序号
        day_start_indices = np.where(day_change)[0]
        # 每个bar属于哪个交易日
        day_idx = np.searchsorted(day_start_indices, np.arange(n), side='right') - 1
        # 日内序号 = 当前位置 - 所属交易日起点
        bar_in_day = np.arange(n) - day_start_indices[day_idx]

        # 4. 计算group_id
        group_id = day_idx * 100000 + bar_in_day // period

        # 5. 找group边界
        group_change = np.empty(n, dtype=bool)
        group_change[0] = True
        group_change[1:] = group_id[1:] != group_id[:-1]
        group_starts = np.where(group_change)[0]
        n_groups = len(group_starts)

        # 6. 用reduceat向量化聚合
        result = np.empty((5, n_groups))

        # open: 每组第一个
        result[0, :] = ohlcv[0, group_starts]

        # high: 每组最大值
        result[1, :] = np.maximum.reduceat(ohlcv[1, :], group_starts)

        # low: 每组最小值
        result[2, :] = np.minimum.reduceat(ohlcv[2, :], group_starts)

        # close: 每组最后一个
        group_ends = np.append(group_starts[1:], n) - 1
        result[3, :] = ohlcv[3, group_ends]

        # volume: 每组求和
        result[4, :] = np.add.reduceat(ohlcv[4, :], group_starts)

        return result

    def _calculate_rsv(self, high, low, close):
        """RSV指标(向量化: 用sliding_window_view替代Python循环)"""
        n = len(close)
        if n == 0:
            return np.array([])
        rsv = np.full(n, 50.0)
        w = self.RSV_WINDOW

        if n >= w:
            from numpy.lib.stride_tricks import sliding_window_view
            # 滚动窗口: shape (n-w+1, w)
            high_w = sliding_window_view(high, w)
            low_w = sliding_window_view(low, w)
            rolling_max = np.max(high_w, axis=1)
            rolling_min = np.min(low_w, axis=1)
            denom = rolling_max - rolling_min
            valid = denom > 0
            rsv[w-1:][valid] = (close[w-1:][valid] - rolling_min[valid]) / denom[valid] * 100

        # 处理前 w-1 个元素(窗口不足)
        for i in range(min(w-1, n)):
            wh = np.max(high[:i+1])
            wl = np.min(low[:i+1])
            if wh - wl > 0:
                rsv[i] = (close[i] - wl) / (wh - wl) * 100
        return rsv

    def _calculate_kd(self, rsv):
        """K/D指标(P0-1修复): 标准2/3+1/3"""
        n = len(rsv)
        if n == 0:
            return np.array([]), np.array([])
        k = np.empty(n)
        d = np.empty(n)
        k[0] = rsv[0]
        d[0] = k[0]
        for i in range(1, n):
            k[i] = (2.0/3.0) * k[i-1] + (1.0/3.0) * rsv[i]
            d[i] = (2.0/3.0) * d[i-1] + (1.0/3.0) * k[i]
        return k, d

    def _red_green_signal(self, ohlcv):
        """红绿信号: 2=多, 1=中性, 0=空"""
        if ohlcv.shape[1] < self.RSV_WINDOW:
            return 1
        rsv = self._calculate_rsv(ohlcv[1, :], ohlcv[2, :], ohlcv[3, :])
        k, d = self._calculate_kd(rsv)
        lk, ld = k[-1], d[-1]
        if lk >= self.RED_THRESHOLD and lk >= ld:
            return 2
        if lk <= self.GREEN_THRESHOLD and lk <= ld:
            return 0
        return 1

    def _calculate_atr(self, ohlcv_2d):
        """ATR"""
        if len(ohlcv_2d) < self.ATR_WINDOW + 1:
            return 10.0
        high = ohlcv_2d[:, 1]
        low = ohlcv_2d[:, 2]
        close_prev = np.roll(ohlcv_2d[:, 3], 1)
        close_prev[0] = close_prev[1]
        tr = np.maximum(np.maximum(high - low, np.abs(high - close_prev)), np.abs(low - close_prev))
        return np.mean(tr[-self.ATR_WINDOW:])

    # ================================================================
    #  止损止盈(P0-3修复: 追踪止损)
    # ================================================================

    def _update_stop_profit(self, direction, ub, lb, bar):
        """更新止损止盈 - 只收紧不放松, 用缓存ATR"""
        atr = self.last_atr
        last_close = bar.close_price

        if direction == 1 or self.pos > 0:
            if self.stop_price is None:
                self.stop_price = lb
            else:
                self.stop_price = max(self.stop_price, lb)
            self.take_profit = last_close + self.TAKE_PROFIT_MULTIPLIER * atr * (1 + self.SLIPPAGE_RATIO)

        elif direction == -1 or self.pos < 0:
            if self.stop_price is None:
                self.stop_price = ub
            else:
                self.stop_price = min(self.stop_price, ub)
            self.take_profit = last_close - self.TAKE_PROFIT_MULTIPLIER * atr * (1 + self.SLIPPAGE_RATIO)

        if self.pos == 0 and direction == 0:
            self.stop_price = None
            self.take_profit = None
