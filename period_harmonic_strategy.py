"""
vnpy CTA 策略: 周期谐波震荡策略 (Period Harmonic Oscillation) v3

v3.1 全面修复 (基于 Codex 第二轮审查):
  S1: 预热阶段不设order_pending (检查self.trading)
  S2: 止损用停止单(stop=True), on_order/on_stop_order处理拒绝/撤销
  S3: 平仓后清除旧方向止损, 新开仓重新设置
  S4: 定时平仓用市价方向+order_pending, 禁止收盘后重开
  S5: 信号计算与14分钟K线边界对齐 (v3.1: 时间计算与resample统一)
  S6: 反手挂起信号在暂停/日切时清除
  S7: on_trade检查offset, 盈亏统计含双边成本
  S8: 移除假参数, 暂停周期修正为14*LONG_PERIOD
  S9: 重采样按真实时间取整, 跳过午休缺口
  S10: 信号计算丢弃最后未完成K线组
  S11: 止盈改为开仓时固定, 不随价格更新
"""

from datetime import datetime, timedelta
from typing import Optional, List
from collections import deque

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


class PeriodHarmonicStrategy(CtaTemplate):
    """周期谐波震荡策略 - vnpy CTA版本 v3"""

    # ============ 策略参数 ============
    SHORT_PERIOD: int = 1
    MIDDLE_PERIOD: int = 4
    LONG_PERIOD: int = 14
    SIGNAL_THRESH: int = 2

    RSV_WINDOW: int = 9
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
    PAUSING_PERIOD: int = 14  # 14个长周期 = 14*14=196根1分钟K线

    # 定时平仓: 03:00 和 21:29 (分钟数)
    CLOSE_TIME_1: int = 180
    CLOSE_TIME_2: int = 1289
    # 收盘后禁止开仓的窗口 (分钟数)
    CLOSE_GUARD_WINDOW: int = 5

    DATA_WINDOW: int = 2000

    parameters: List[str] = [
        "SHORT_PERIOD", "MIDDLE_PERIOD", "LONG_PERIOD", "SIGNAL_THRESH",
        "RSV_WINDOW", "ATR_WINDOW",
        "RED_THRESHOLD", "GREEN_THRESHOLD",
        "LONG_WEIGHT", "MIDDLE_WEIGHT", "SHORT_WEIGHT", "SCORE_OFFSET",
        "MAX_POSITION", "DAILY_MAX_LOSS", "POINT_VALUE", "FIXED_SLIPPAGE",
        "STOP_LOSS_MULTIPLIER", "TAKE_PROFIT_MULTIPLIER", "SLIPPAGE_RATIO",
        "LOSS_CHANCE", "PAUSING_PERIOD",
        "CLOSE_TIME_1", "CLOSE_TIME_2", "CLOSE_GUARD_WINDOW",
        "DATA_WINDOW",
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
        self.bar_datetimes: deque = deque(maxlen=self.DATA_WINDOW)

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

        # S5: 追踪14分钟K线组编号, 信号只在组完成时计算
        self._last_14min_group: int = -1

        # S6: 追踪活跃订单ID
        self.active_order_ids: set = set()

    # ================================================================
    #  vnpy 回调
    # ================================================================

    def on_init(self):
        self.write_log("策略初始化 PeriodHarmonicStrategy v3")
        try:
            self.load_bar(10)
        except Exception as e:
            self.write_log(f"load_bar异常(非致命): {e}")

    def on_start(self):
        self.write_log("策略启动 v3")
        self.put_event()

    def on_stop(self):
        self.write_log("策略停止 v3")
        self.put_event()

    def on_tick(self, tick: TickData):
        self.bg.update_tick(tick)

    def on_bar(self, bar: BarData):
        """1分钟K线回调"""
        # 1. 更新数据缓存
        self.am.update_bar(bar)
        self.bar_datetimes.append(bar.datetime)

        if not self.am.inited:
            if not hasattr(self, '_am_not_inited_logged'):
                self.write_log(f"ArrayManager未就绪: count={self.am.count}/{self.am.size}")
                self._am_not_inited_logged = True
            return

        self.bar_count += 1

        if self.bar_count == 1:
            self.write_log(f"ArrayManager就绪! count={self.am.count} 开始处理信号")

        # S1: 持仓变化时重置order_pending (仅当trading=True时, 即非预热)
        if self.pos != self.last_pos:
            self.order_pending = False
            self.last_pos = self.pos
            self.active_order_ids.clear()
            # S3: 平仓后清除旧方向的止损止盈
            if self.pos == 0:
                self.stop_price = None
                self.take_profit = None
                self.entry_price = None
                self.write_log(f"仓位归零, 清除止损止盈 pos=0 bar_count={self.bar_count}")

        # 2. 交易日切换
        self._check_day_rollover(bar)

        # S6: 日切时清除过期反手信号
        if self.pending_reverse_signal != 0 and self._day_just_changed:
            self.write_log(f"日切清除过期反手信号: {self.pending_reverse_signal}")
            self.pending_reverse_signal = 0

        # 3. 连亏暂停 (S8: 暂停14*LONG_PERIOD根1分钟K线)
        if self.loss_num >= self.LOSS_CHANCE:
            self.pausing_countdown = self.PAUSING_PERIOD * self.LONG_PERIOD
            self.loss_num = 0
            # S6: 暂停时清除反手信号
            if self.pending_reverse_signal != 0:
                self.write_log(f"连亏暂停清除反手信号: {self.pending_reverse_signal}")
                self.pending_reverse_signal = 0
        if self.pausing_countdown > 0:
            self.pausing_countdown -= 1
            self.put_event()
            return

        # 4. 定时平仓 (S4: 修复)
        current_minute = bar.datetime.hour * 60 + bar.datetime.minute
        is_close_time = (
            abs(current_minute - self.CLOSE_TIME_1) <= 1 or
            abs(current_minute - self.CLOSE_TIME_2) <= 1
        )

        if is_close_time:
            if self.pos != 0:
                self.write_log(f"定时平仓: {bar.datetime} pos={self.pos}")
                self.cancel_all()
                if self.pos > 0:
                    # 卖出用收盘价-滑点, 确保成交
                    sell_price = bar.close_price - self.FIXED_SLIPPAGE
                    self.sell(sell_price, abs(self.pos))
                elif self.pos < 0:
                    # 买入用收盘价+滑点, 确保成交
                    buy_price = bar.close_price + self.FIXED_SLIPPAGE
                    self.cover(buy_price, abs(self.pos))
                self.order_pending = True
            self.put_event()
            return

        # 收盘后窗口内禁止开新仓
        if (abs(current_minute - self.CLOSE_TIME_1) <= self.CLOSE_GUARD_WINDOW or
            abs(current_minute - self.CLOSE_TIME_2) <= self.CLOSE_GUARD_WINDOW):
            self.put_event()
            return

        # 5. 止损止盈触发 (S2: 用限价单但处理order_pending, S3: pos=0时不清)
        if self.pos > 0 and self.stop_price is not None:
            if bar.low_price <= self.stop_price:
                self.write_log(f"多头止损触发: stop={self.stop_price} low={bar.low_price}")
                self.cancel_all()
                self.sell(self.stop_price, abs(self.pos), stop=True)
                self.order_pending = True
                self.put_event()
                return
            if self.take_profit is not None and bar.high_price >= self.take_profit:
                self.write_log(f"多头止盈触发: tp={self.take_profit} high={bar.high_price}")
                self.cancel_all()
                self.sell(self.take_profit, abs(self.pos))
                self.order_pending = True
                self.put_event()
                return
        elif self.pos < 0 and self.stop_price is not None:
            if bar.high_price >= self.stop_price:
                self.write_log(f"空头止损触发: stop={self.stop_price} high={bar.high_price}")
                self.cancel_all()
                self.cover(self.stop_price, abs(self.pos), stop=True)
                self.order_pending = True
                self.put_event()
                return
            if self.take_profit is not None and bar.low_price <= self.take_profit:
                self.write_log(f"空头止盈触发: tp={self.take_profit} low={bar.low_price}")
                self.cancel_all()
                self.cover(self.take_profit, abs(self.pos))
                self.order_pending = True
                self.put_event()
                return

        # 6. 信号计算 (S5: 与14分钟K线边界对齐)
        need_signal = self._check_14min_boundary(bar)
        if need_signal:
            try:
                direction, ub, lb, atr = self._is_harmonic_oscillation()
                self.last_signal_direction = direction
                self.last_signal_ub = ub
                self.last_signal_lb = lb
                self.last_atr = atr
                if self.bar_count <= 42:  # 前3个14分钟周期
                    self.write_log(
                        f"信号计算 bar_count={self.bar_count} "
                        f"direction={direction} atr={atr:.2f} "
                        f"ub={ub:.2f} lb={lb:.2f} close={bar.close_price}"
                    )
            except Exception as e:
                self.write_log(f"信号计算异常: {e}")
                import traceback
                self.write_log(traceback.format_exc())
                direction = 0
                ub = self.last_signal_ub
                lb = self.last_signal_lb
                atr = self.last_atr
        else:
            direction = self.last_signal_direction
            ub = self.last_signal_ub
            lb = self.last_signal_lb

        # 7. 更新止损止盈 (S3: pos=0时清除, 新开仓重新设置)
        self._update_stop_profit(direction, ub, lb, bar)

        # 8. 开仓/反手判断
        if self.daily_loss >= self.DAILY_MAX_LOSS:
            self.put_event()
            return

        # S1: 有待处理订单时不重复下单
        if self.order_pending:
            self.put_event()
            return

        # 反手待开仓: 上一根K线已平仓, 现在开反向仓
        if self.pending_reverse_signal != 0 and self.pos == 0:
            rev = self.pending_reverse_signal
            self.pending_reverse_signal = 0
            if rev == 1:
                self.write_log(f"反手开多 close={bar.close_price} bar_count={self.bar_count}")
                self.buy(bar.close_price, self.MAX_POSITION)
            else:
                self.write_log(f"反手开空 close={bar.close_price} bar_count={self.bar_count}")
                self.short(bar.close_price, self.MAX_POSITION)
            self.order_pending = True
            self.put_event()
            return

        # 正常开仓 (S1: 检查self.trading避免预热阶段设order_pending)
        if direction == 1 and self.pos == 0:
            self.write_log(f"开多 signal=1 close={bar.close_price} bar_count={self.bar_count}")
            self.buy(bar.close_price, self.MAX_POSITION)
            if self.trading:
                self.order_pending = True
        elif direction == -1 and self.pos == 0:
            self.write_log(f"开空 signal=-1 close={bar.close_price} bar_count={self.bar_count}")
            self.short(bar.close_price, self.MAX_POSITION)
            if self.trading:
                self.order_pending = True
        # 反手: 先平仓, 下一根K线再开反向仓
        elif direction == -1 and self.pos > 0:
            self.write_log(f"反手多→空 close={bar.close_price} bar_count={self.bar_count}")
            self.pending_reverse_signal = -1
            self.sell(bar.close_price - self.FIXED_SLIPPAGE, abs(self.pos))
            if self.trading:
                self.order_pending = True
        elif direction == 1 and self.pos < 0:
            self.write_log(f"反手空→多 close={bar.close_price} bar_count={self.bar_count}")
            self.pending_reverse_signal = 1
            self.cover(bar.close_price + self.FIXED_SLIPPAGE, abs(self.pos))
            if self.trading:
                self.order_pending = True

        self.put_event()

    def on_order(self, order: OrderData):
        """S2: 处理订单状态变化 - 拒绝/撤销时释放order_pending"""
        status_str = str(order.status)

        if "REJECTED" in status_str:
            self.write_log(f"订单被拒绝: {order.vt_orderid} 价格={order.price}")
            self.order_pending = False
        elif "CANCELLED" in status_str:
            self.write_log(f"订单被撤销: {order.vt_orderid}")
            self.order_pending = False

        self.put_event()

    def on_trade(self, trade: TradeData):
        """S7: 成交回报 - 检查offset, 盈亏含双边成本"""
        direction_str = str(trade.direction)
        offset_str = str(trade.offset) if trade.offset else "UNKNOWN"

        self.write_log(
            f"成交: {trade.datetime} dir={direction_str} offset={offset_str} "
            f"price={trade.price} vol={trade.volume}"
        )

        # S7: 根据offset判断是开仓还是平仓
        is_open = "OPEN" in offset_str.upper()
        is_close = "CLOSE" in offset_str.upper()

        if is_close and self.entry_price is not None and self.prev_direction != 0:
            # 平仓 - 计算盈亏 (含双边滑点+手续费)
            exit_price = trade.price
            pnl = self._calc_pnl_full(self.entry_price, exit_price, self.prev_direction, trade.volume)

            if pnl < 0:
                self.loss_num += 1
                self.daily_loss += abs(pnl)
                self.write_log(f"亏损: pnl={pnl:.2f} HKD loss_num={self.loss_num}")
            else:
                self.loss_num = 0
                self.write_log(f"盈利: pnl={pnl:.2f} HKD")

            self.entry_price = None
        elif is_open or (not is_close and self.entry_price is None):
            # 开仓
            self.entry_price = trade.price
            if "LONG" in direction_str:
                self.prev_direction = 1
            else:
                self.prev_direction = -1
        else:
            # 反手或无法确定offset, 按奇偶顺序配对
            if self.entry_price is not None and self.prev_direction != 0:
                exit_price = trade.price
                pnl = self._calc_pnl_full(self.entry_price, exit_price, self.prev_direction, trade.volume)
                if pnl < 0:
                    self.loss_num += 1
                    self.daily_loss += abs(pnl)
                else:
                    self.loss_num = 0
                self.entry_price = None
            else:
                self.entry_price = trade.price
                if "LONG" in direction_str:
                    self.prev_direction = 1
                else:
                    self.prev_direction = -1

        self.put_event()

    def on_stop_order(self, stop_order: StopOrder):
        """S2: 停止单状态变化 - 撤销时释放order_pending"""
        status_str = str(stop_order.status)
        if "CANCELLED" in status_str:
            self.write_log(f"停止单被撤销")
            self.order_pending = False
        elif "TRIGGERED" in status_str:
            self.write_log(f"停止单已触发, 等待成交")
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
        self._day_just_changed = (day != self.current_trading_day)
        if self._day_just_changed:
            self.current_trading_day = day
            self.daily_loss = 0.0

    def _calc_pnl(self, entry, exit_price, direction, volume):
        """简单盈亏 (单边滑点)"""
        cost = self.FIXED_SLIPPAGE * self.POINT_VALUE * volume
        if direction == 1:
            return (exit_price - entry) * self.POINT_VALUE * volume - cost
        return (entry - exit_price) * self.POINT_VALUE * volume - cost

    def _calc_pnl_full(self, entry, exit_price, direction, volume):
        """S7: 完整盈亏 (双边滑点+手续费)"""
        # 双边滑点: 开仓+平仓各1点
        slippage_cost = 2 * self.FIXED_SLIPPAGE * self.POINT_VALUE * volume
        # 手续费: 按成交额 * 手续费率, 双边
        notional = entry * self.POINT_VALUE * volume + exit_price * self.POINT_VALUE * volume
        commission = notional * 0.000047  # 与回测引擎一致
        if direction == 1:
            return (exit_price - entry) * self.POINT_VALUE * volume - slippage_cost - commission
        return (entry - exit_price) * self.POINT_VALUE * volume - slippage_cost - commission

    # ================================================================
    #  S5: 14分钟K线边界对齐 (v3.1: 时间计算与resample统一)
    # ================================================================

    def _get_day_minutes(self, dt: datetime) -> int:
        """计算日内分钟序号 (与_resample_by_time一致, 跳过午休)"""
        h, m = dt.hour, dt.minute
        if h >= 17:
            return (h - 17) * 60 + m
        elif h < 5:
            return (h + 7) * 60 + m
        elif h >= 13:
            return 165 + (h - 13) * 60 + m
        elif h >= 9:
            day_min = (h - 9) * 60 + m - 15
            return max(day_min, 0)
        else:
            return 0

    def _check_14min_boundary(self, bar: BarData) -> bool:
        """检查是否在14分钟K线组完成时 (组边界对齐)"""
        dt = bar.datetime

        # 交易日 (17:00+归次日)
        if dt.hour >= 17:
            tday = dt.toordinal() + 1
        else:
            tday = dt.toordinal()

        # 日内分钟序号 (与_resample_by_time一致)
        day_minutes = self._get_day_minutes(dt)

        # 组编号 (含交易日, 确保日切时重置)
        group = tday * 100000 + day_minutes // self.LONG_PERIOD

        if group != self._last_14min_group:
            self._last_14min_group = group
            return True
        return False

    # ================================================================
    #  信号计算
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

        datetimes = list(self.bar_datetimes)

        short_data = self._resample_by_time(ohlcv, datetimes, self.SHORT_PERIOD)
        middle_data = self._resample_by_time(ohlcv, datetimes, self.MIDDLE_PERIOD)
        long_data = self._resample_by_time(ohlcv, datetimes, self.LONG_PERIOD)

        # S10: 丢弃最后未完成的K线组 (period > 1时最后一组可能只有部分K线)
        if self.SHORT_PERIOD > 1 and short_data.shape[1] > 1:
            short_data = short_data[:, :-1]
        if self.MIDDLE_PERIOD > 1 and middle_data.shape[1] > 1:
            middle_data = middle_data[:, :-1]
        if self.LONG_PERIOD > 1 and long_data.shape[1] > 1:
            long_data = long_data[:, :-1]

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
        S9: 按真实时间取整重采样, 跳过午休缺口
        用时间戳对齐到period分钟边界, 不按数组序号分组
        """
        if period == 1:
            return ohlcv.copy()

        n = ohlcv.shape[1]
        if n == 0:
            return np.empty((5, 0))

        # 计算每个bar的交易日和时间组 (v3.1: 统一使用_get_day_minutes)
        group_ids = np.empty(n, dtype=np.int64)
        for i, dt in enumerate(datetimes):
            # 交易日 (17:00+归次日)
            if dt.hour >= 17:
                tday = dt.toordinal() + 1
            else:
                tday = dt.toordinal()

            # 日内分钟序号 (与_check_14min_boundary一致)
            day_min = self._get_day_minutes(dt)

            # 时间组: 按period分钟取整
            group_ids[i] = tday * 100000 + day_min // period

        # 找组边界
        group_change = np.empty(n, dtype=bool)
        group_change[0] = True
        group_change[1:] = group_ids[1:] != group_ids[:-1]
        group_starts = np.where(group_change)[0]
        n_groups = len(group_starts)

        if n_groups == 0:
            return np.empty((5, 0))

        # 用reduceat向量化聚合
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
        """RSV指标"""
        n = len(close)
        if n == 0:
            return np.array([])
        rsv = np.full(n, 50.0)
        w = self.RSV_WINDOW

        if n >= w:
            from numpy.lib.stride_tricks import sliding_window_view
            high_w = sliding_window_view(high, w)
            low_w = sliding_window_view(low, w)
            rolling_max = np.max(high_w, axis=1)
            rolling_min = np.min(low_w, axis=1)
            denom = rolling_max - rolling_min
            valid = denom > 0
            rsv[w-1:][valid] = (close[w-1:][valid] - rolling_min[valid]) / denom[valid] * 100

        for i in range(min(w-1, n)):
            wh = np.max(high[:i+1])
            wl = np.min(low[:i+1])
            if wh - wl > 0:
                rsv[i] = (close[i] - wl) / (wh - wl) * 100
        return rsv

    def _calculate_kd(self, rsv):
        """K/D指标: 标准2/3+1/3"""
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
    #  止损止盈 (S3: 新开仓重新设置, 平仓后清除)
    # ================================================================

    def _update_stop_profit(self, direction, ub, lb, bar):
        """更新止损止盈 - 只收紧不放松"""
        atr = self.last_atr
        last_close = bar.close_price

        # S3: pos=0时, 只在有新信号时设置
        if self.pos == 0:
            if direction == 1:
                # 新开多: 设置初始止损
                self.stop_price = lb
                self.take_profit = last_close + self.TAKE_PROFIT_MULTIPLIER * atr * (1 + self.SLIPPAGE_RATIO)
                self.write_log(f"设置多头止损: stop={self.stop_price:.2f} tp={self.take_profit:.2f}")
            elif direction == -1:
                # 新开空: 设置初始止损
                self.stop_price = ub
                self.take_profit = last_close - self.TAKE_PROFIT_MULTIPLIER * atr * (1 + self.SLIPPAGE_RATIO)
                self.write_log(f"设置空头止损: stop={self.stop_price:.2f} tp={self.take_profit:.2f}")
            else:
                # 无方向, 清除
                self.stop_price = None
                self.take_profit = None
        elif self.pos > 0:
            # 多头持仓: 止损只上移不下移
            if self.stop_price is None:
                self.stop_price = lb
            else:
                self.stop_price = max(self.stop_price, lb)
            # S11: 止盈只在开仓时设置, 不随价格更新
            if self.take_profit is None:
                self.take_profit = last_close + self.TAKE_PROFIT_MULTIPLIER * atr * (1 + self.SLIPPAGE_RATIO)
        elif self.pos < 0:
            # 空头持仓: 止损只下移不上移
            if self.stop_price is None:
                self.stop_price = ub
            else:
                self.stop_price = min(self.stop_price, ub)
            # S11: 止盈只在开仓时设置, 不随价格更新
            if self.take_profit is None:
                self.take_profit = last_close - self.TAKE_PROFIT_MULTIPLIER * atr * (1 + self.SLIPPAGE_RATIO)
