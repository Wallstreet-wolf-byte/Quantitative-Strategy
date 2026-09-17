"""
vnpy CTA 策略: 周期谐波震荡策略 (Period Harmonic Oscillation) v3.3

v3.2 全面重构 (基于 Codex 第三轮审查):
  三周期分层共振: 14分钟定趋势, 4分钟确认, 1分钟触发入场
  S1: 预热阶段不设order_pending (检查self.trading)
  S2: 止损用停止单(stop=True)预挂, on_stop_order用stop_orderid属性
  S3: 平仓后清除旧方向止损, 新开仓重新设置
  S4: 定时平仓用市价方向+order_pending, 禁止收盘后重开
  S5: 信号与K线边界对齐, 各周期只在完整K线收盘时更新
  S6: 14分钟反转只平仓不反手; 日切/暂停时清理状态
  S7: on_trade检查offset, 盈亏统计含双边成本, 多笔成交累积
  S8: 移除假参数, 暂停周期修正为14*LONG_PERIOD
  S9: 重采样按交易时段独立零点(09:15/13:00/17:15)
  S10: resample内部丢弃不完整K线组
  S11: 止盈改为开仓时固定, 不随价格更新
  S12: 限价单超时2根K线自动撤单
  S13: 14分钟反转平仓优先于风控检查, 不被暂停/亏损拦截
  S14: KD/ATR按新K线增量更新, 不再每分钟重算整个历史窗口
  S15: backtest_hsi.py用Direction/Offset枚举, 修复语法错误
  S16: 开仓/退出委托分别追踪, 支持部分成交和退出超时重试
  S17: 保护单数量随实际持仓同步
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
from vnpy.trader.constant import Direction, Offset, Status


class PeriodHarmonicStrategy(CtaTemplate):
    """周期谐波震荡策略 - vnpy CTA版本 v3.3"""

    # ============ 策略参数 ============
    SHORT_PERIOD: int = 1
    MIDDLE_PERIOD: int = 4
    LONG_PERIOD: int = 14

    RSV_WINDOW: int = 9
    ATR_WINDOW: int = 14

    RED_THRESHOLD: float = 80.0
    GREEN_THRESHOLD: float = 20.0

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
        "SHORT_PERIOD", "MIDDLE_PERIOD", "LONG_PERIOD",
        "RSV_WINDOW", "ATR_WINDOW",
        "RED_THRESHOLD", "GREEN_THRESHOLD",
        "MAX_POSITION", "DAILY_MAX_LOSS", "POINT_VALUE", "FIXED_SLIPPAGE",
        "STOP_LOSS_MULTIPLIER", "TAKE_PROFIT_MULTIPLIER", "SLIPPAGE_RATIO",
        "LOSS_CHANCE", "PAUSING_PERIOD",
        "CLOSE_TIME_1", "CLOSE_TIME_2", "CLOSE_GUARD_WINDOW",
        "DATA_WINDOW",
    ]

    variables: List[str] = [
        "loss_num", "pausing_countdown", "daily_loss",
        "stop_price", "take_profit", "current_trading_day",
        "entry_price", "entry_volume",
        "short_direction", "middle_direction", "long_direction",
        "prev_short_direction",
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
        self.entry_volume: int = 0
        self.prev_direction: int = 0

        # 三周期方向
        self.short_direction: int = 0
        self.middle_direction: int = 0
        self.long_direction: int = 0
        self.prev_short_direction: int = 0

        self.last_atr: float = 10.0
        self.bar_count: int = 0
        self.order_pending: bool = False
        self.last_pos: int = 0

        # 组编号追踪
        self._last_4min_group: int = -1
        self._last_14min_group: int = -1

        # S14: 增量KD/ATR状态
        self._indicator_inited: bool = False
        self._period_states: dict = {}
        self._atr_values: deque = deque(maxlen=self.ATR_WINDOW)
        self._previous_close: Optional[float] = None

        # 订单ID追踪
        self.stop_orderid: str = ""
        self.tp_orderid: str = ""
        self.entry_orderids: set = set()
        self.exit_orderids: set = set()
        self.entry_order_placed_bar: int = 0
        self.exit_order_placed_bar: int = 0
        self.force_exit_reason: str = ""
        self.protected_volume: int = 0
        self.protection_sync_pending: bool = False
        self.stop_replace_pending: bool = False
        self.protection_cancel_ids: set = set()

        self._day_just_changed: bool = False

    # ================================================================
    #  vnpy 回调
    # ================================================================

    def on_init(self):
        self.write_log("策略初始化 PeriodHarmonicStrategy v3.3")
        try:
            self.load_bar(10)
        except Exception as e:
            self.write_log(f"load_bar异常(非致命): {e}")

    def on_start(self):
        self.write_log("策略启动 v3.3")
        self.put_event()

    def on_stop(self):
        self.write_log("策略停止 v3.3")
        self.put_event()

    def on_tick(self, tick: TickData):
        self.bg.update_tick(tick)

    def on_bar(self, bar: BarData):
        """1分钟K线回调"""
        # 0. 更新数据缓存
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

        # S12/S16: 开仓和强制退出委托都做超时管理
        if self.entry_orderids and self.bar_count - self.entry_order_placed_bar >= 2:
            self.write_log(
                f"开仓单超时撤单: bar_count={self.bar_count} "
                f"placed={self.entry_order_placed_bar}"
            )
            for vt_orderid in list(self.entry_orderids):
                self.cancel_order(vt_orderid)

        if self.exit_orderids and self.bar_count - self.exit_order_placed_bar >= 2:
            self.write_log(
                f"退出单超时撤单并重试: bar_count={self.bar_count} "
                f"placed={self.exit_order_placed_bar}"
            )
            for vt_orderid in list(self.exit_orderids):
                self.cancel_order(vt_orderid)

        self._refresh_order_pending()

        # 1. 持仓变化检测
        if self.pos != self.last_pos:
            self.last_pos = self.pos

            if self.pos == 0:
                # S3: 仓位归零, 清除止损止盈, 撤销所有残留订单
                self.stop_price = None
                self.take_profit = None
                self.entry_price = None
                self.entry_volume = 0
                self.protected_volume = 0
                self.protection_sync_pending = False
                self.stop_replace_pending = False
                self.force_exit_reason = ""
                self._cancel_protective_orders()
                self.cancel_all()
                self._refresh_order_pending()
                self.write_log(f"仓位归零, 清除止损止盈 pos=0 bar_count={self.bar_count}")
            elif self.pos != 0 and self.stop_price is not None:
                self._request_protection_sync()

        # 2. 交易日切换
        self._check_day_rollover(bar)

        # S6: 日切时清除状态
        if self._day_just_changed:
            if self.long_direction != 0 or self.middle_direction != 0:
                self.write_log(f"日切: long_dir={self.long_direction} mid_dir={self.middle_direction}")
            self.prev_short_direction = 0

        # 指标始终连续更新, 包括暂停和强制退出期间。
        self._update_signals(bar)
        self._try_place_protective_orders()
        self._try_replace_stop_order()

        # 3. 连亏暂停触发 (只设倒计时, 不return)
        if self.loss_num >= self.LOSS_CHANCE:
            self.pausing_countdown = self.PAUSING_PERIOD * self.LONG_PERIOD
            self.loss_num = 0
            self.write_log(f"连亏暂停 {self.pausing_countdown} 根K线")

        # 未完成的强制退出优先重试, 不允许仓位失去退出追踪
        if self.force_exit_reason and self.pos != 0:
            if not self.exit_orderids:
                self._submit_forced_exit(bar, self.force_exit_reason)
            self.put_event()
            return

        # 4. 定时平仓 (提前一根K线发单, 让最后一根K线完成撮合)
        current_minute = bar.datetime.hour * 60 + bar.datetime.minute
        scheduled_exit_minutes = (self.CLOSE_TIME_1 - 2, self.CLOSE_TIME_2 - 1)
        is_close_time = current_minute in scheduled_exit_minutes

        if is_close_time:
            if self.pos != 0:
                self.write_log(f"定时平仓: {bar.datetime} pos={self.pos}")
                self._submit_forced_exit(bar, "scheduled")
            self.put_event()
            return

        # 收盘后窗口内禁止开新仓 (但允许已有仓位管理)
        in_close_guard = (
            abs(current_minute - self.CLOSE_TIME_1) <= self.CLOSE_GUARD_WINDOW or
            abs(current_minute - self.CLOSE_TIME_2) <= self.CLOSE_GUARD_WINDOW
        )

        # 5. 更新止损止盈 (S11: 止盈只在开仓时设置)
        # 止损预挂单模式下, 止损上移时更新停止单
        if self.pos != 0 and self.stop_price is not None:
            self._update_trailing_stop(bar)

        # 7a. 14分钟反转平仓 (优先于暂停/亏损/order_pending, 确保持仓能退出)
        if self.pos > 0 and self.long_direction == -1:
            self.write_log(f"14分钟转空, 平多仓 close={bar.close_price} bar_count={self.bar_count}")
            self._submit_forced_exit(bar, "14m_reversal")
            self.put_event()
            return

        elif self.pos < 0 and self.long_direction == 1:
            self.write_log(f"14分钟转多, 平空仓 close={bar.close_price} bar_count={self.bar_count}")
            self._submit_forced_exit(bar, "14m_reversal")
            self.put_event()
            return

        # 7b. 连亏暂停 (在反转平仓之后, 不阻止仓位退出)
        if self.pausing_countdown > 0:
            self.pausing_countdown -= 1
            self.put_event()
            return

        # 7c. 收盘后窗口禁止开新仓
        if in_close_guard:
            self.put_event()
            return

        # 7d. 每日亏损限制 (在反转平仓之后, 不阻止仓位退出)
        if self.daily_loss >= self.DAILY_MAX_LOSS:
            self.put_event()
            return

        # 7e. 订单挂起检查
        if self.order_pending:
            self.put_event()
            return

        # 8. 入场判断
        # 计算趋势和入场信号
        trend = self.long_direction if (self.long_direction == self.middle_direction != 0) else 0
        entry_signal = 0
        if trend != 0:
            if self.short_direction == trend and self.prev_short_direction != trend:
                entry_signal = trend

        # 8a. 入场 (空仓 + entry_signal)
        if entry_signal == 1 and self.pos == 0:
            self.write_log(
                f"开多 entry_signal=1 close={bar.close_price} bar_count={self.bar_count} "
                f"long={self.long_direction} mid={self.middle_direction} "
                f"short={self.short_direction} prev_short={self.prev_short_direction}"
            )
            # 设置止损止盈 (在开仓前设置, 成交后预挂)
            atr = self.last_atr
            self.stop_price = bar.close_price - self.STOP_LOSS_MULTIPLIER * atr * (1 + self.SLIPPAGE_RATIO)
            self.take_profit = bar.close_price + self.TAKE_PROFIT_MULTIPLIER * atr * (1 + self.SLIPPAGE_RATIO)
            self.prev_direction = 1
            ids = self.buy(bar.close_price, self.MAX_POSITION)
            if self.trading and ids:
                self.entry_orderids.update(ids)
                self.entry_order_placed_bar = self.bar_count
                self._refresh_order_pending()

        elif entry_signal == -1 and self.pos == 0:
            self.write_log(
                f"开空 entry_signal=-1 close={bar.close_price} bar_count={self.bar_count} "
                f"long={self.long_direction} mid={self.middle_direction} "
                f"short={self.short_direction} prev_short={self.prev_short_direction}"
            )
            atr = self.last_atr
            self.stop_price = bar.close_price + self.STOP_LOSS_MULTIPLIER * atr * (1 + self.SLIPPAGE_RATIO)
            self.take_profit = bar.close_price - self.TAKE_PROFIT_MULTIPLIER * atr * (1 + self.SLIPPAGE_RATIO)
            self.prev_direction = -1
            ids = self.short(bar.close_price, self.MAX_POSITION)
            if self.trading and ids:
                self.entry_orderids.update(ids)
                self.entry_order_placed_bar = self.bar_count
                self._refresh_order_pending()

        self.put_event()

    def on_order(self, order: OrderData):
        """处理限价委托状态, 保留部分成交委托的追踪。"""
        order_id = order.vt_orderid

        if order.status == Status.REJECTED:
            self.write_log(f"订单被拒绝: {order_id} 价格={order.price}")

        elif order.status == Status.CANCELLED:
            self.write_log(f"订单被撤销: {order_id}")

        if not order.is_active():
            self.entry_orderids.discard(order_id)
            self.exit_orderids.discard(order_id)
            self.protection_cancel_ids.discard(order_id)

        # 止盈是普通限价单, 全成后撤销止损单。
        if order.status == Status.ALLTRADED and order_id == self.tp_orderid:
            self.tp_orderid = ""
            self.protection_sync_pending = False
            self.stop_replace_pending = False
            if self.stop_orderid:
                stop_orderid = self.stop_orderid
                self.stop_orderid = ""
                self.cancel_order(stop_orderid)
        elif not order.is_active() and order_id == self.tp_orderid:
            self.tp_orderid = ""
            if order.status == Status.REJECTED and self.pos != 0 and not self.force_exit_reason:
                self.protection_sync_pending = True

        self._try_place_protective_orders()
        self._try_replace_stop_order()

        self._refresh_order_pending()

        self.put_event()

    def on_trade(self, trade: TradeData):
        """S7: 成交回报 - 多笔成交支持 + 枚举比较"""
        self.write_log(
            f"成交: {trade.datetime} dir={trade.direction} offset={trade.offset} "
            f"price={trade.price} vol={trade.volume}"
        )

        is_open = (trade.offset == Offset.OPEN)
        is_close = trade.offset in (Offset.CLOSE, Offset.CLOSETODAY, Offset.CLOSEYESTERDAY)

        if is_open:
            # 开仓成交: 累积加权均价
            if self.entry_volume > 0 and self.entry_price is not None:
                total_cost = self.entry_price * self.entry_volume + trade.price * trade.volume
                self.entry_volume += trade.volume
                self.entry_price = total_cost / self.entry_volume
            else:
                self.entry_price = trade.price
                self.entry_volume = trade.volume
            # 委托ID在on_order中按成交状态移除。这里不清空集合,
            # 这样部分成交的剩余数量仍会继续被超时检查。
            self._request_protection_sync()

        elif is_close and self.entry_price is not None and self.prev_direction != 0:
            # 一旦开始平仓, 立即撤销尚未成交的开仓余单，防止保护单成交后又被补回仓位。
            for vt_orderid in list(self.entry_orderids):
                self.cancel_order(vt_orderid)

            # 平仓成交: 计算本次平仓部分盈亏
            pnl = self._calc_pnl_full(self.entry_price, trade.price, self.prev_direction, trade.volume)

            if pnl < 0:
                self.loss_num += 1
                self.daily_loss += abs(pnl)
                self.write_log(f"亏损: pnl={pnl:.2f} HKD loss_num={self.loss_num}")
            else:
                self.loss_num = 0
                self.write_log(f"盈利: pnl={pnl:.2f} HKD")

            # 减少开仓手数
            self.entry_volume -= trade.volume
            if self.entry_volume <= 0:
                self.entry_price = None
                self.entry_volume = 0

            # S4: 平仓成交时撤销另一侧订单
            self._cancel_opposite_orders(trade)

            if self.pos == 0:
                self.force_exit_reason = ""

        self._refresh_order_pending()

        self.put_event()

    def on_stop_order(self, stop_order: StopOrder):
        """S2: 停止单状态变化"""
        # vn.py 4.4 StopOrder 属性是 stop_orderid, 不是 vt_orderid
        if stop_order.status.name == "CANCELLED":
            self.write_log(f"停止单被撤销")
            was_expected = stop_order.stop_orderid in self.protection_cancel_ids
            was_current = stop_order.stop_orderid == self.stop_orderid
            self.protection_cancel_ids.discard(stop_order.stop_orderid)
            if was_current:
                self.stop_orderid = ""
            if was_current and not was_expected and self.pos != 0 and not self.force_exit_reason:
                self.stop_replace_pending = True
        elif stop_order.status.name == "TRIGGERED":
            self.write_log(f"停止单已触发, 等待成交")
            self.protection_cancel_ids.discard(stop_order.stop_orderid)
            self.protection_sync_pending = False
            self.stop_replace_pending = False
            if stop_order.stop_orderid == self.stop_orderid:
                self.stop_orderid = ""
            # 止损触发 → 撤销止盈单
            if self.tp_orderid:
                tp_orderid = self.tp_orderid
                self.tp_orderid = ""
                self.cancel_order(tp_orderid)

            # vn.py会把停止单转换成普通平仓单, 这些ID必须纳入退出追踪。
            self.exit_orderids.update(stop_order.vt_orderids)
            self.exit_order_placed_bar = self.bar_count

        self._try_place_protective_orders()
        self._try_replace_stop_order()
        self._refresh_order_pending()

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

    def _calc_pnl_full(self, entry, exit_price, direction, volume):
        """S7: 完整盈亏 (双边滑点+手续费)"""
        slippage_cost = 2 * self.FIXED_SLIPPAGE * self.POINT_VALUE * volume
        notional = entry * self.POINT_VALUE * volume + exit_price * self.POINT_VALUE * volume
        commission = notional * 0.000047
        if direction == 1:
            return (exit_price - entry) * self.POINT_VALUE * volume - slippage_cost - commission
        return (entry - exit_price) * self.POINT_VALUE * volume - slippage_cost - commission

    # ================================================================
    #  信号计算: 三周期分层共振
    # ================================================================

    def _update_signals(self, bar: BarData):
        """逐根更新1分钟信号, 仅在完整周期结束后更新4/14分钟信号。"""
        if not self._indicator_inited:
            self._bootstrap_indicators()
            return

        self.prev_short_direction = self.short_direction
        self.short_direction, _ = self._advance_period(bar, self.SHORT_PERIOD)

        new_middle, middle_updated = self._advance_period(bar, self.MIDDLE_PERIOD)
        if middle_updated and new_middle != self.middle_direction:
            self.write_log(
                f"4分钟信号更新: {self.middle_direction}→{new_middle} bar_count={self.bar_count}"
            )
        self.middle_direction = new_middle

        new_long, long_updated = self._advance_period(bar, self.LONG_PERIOD)
        if long_updated and new_long != self.long_direction:
            self.write_log(
                f"14分钟信号更新: {self.long_direction}→{new_long} bar_count={self.bar_count}"
            )
        self.long_direction = new_long

        self._advance_atr(bar)

    def _bootstrap_indicators(self):
        """用ArrayManager中的预热数据初始化一次, 后续全部增量计算。"""
        ohlcv = np.array([
            self.am.open_array,
            self.am.high_array,
            self.am.low_array,
            self.am.close_array,
            self.am.volume_array,
        ])
        datetimes = list(self.bar_datetimes)

        for period in (self.SHORT_PERIOD, self.MIDDLE_PERIOD, self.LONG_PERIOD):
            state = self._new_period_state()
            completed = self._resample_by_time(ohlcv, datetimes, period)
            for i in range(completed.shape[1]):
                self._advance_kd_state(
                    state,
                    completed[1, i],
                    completed[2, i],
                    completed[3, i],
                )

            if period > 1 and datetimes:
                latest_key = self._get_period_key(datetimes[-1], period)
                tail_start = len(datetimes) - 1
                while (
                    tail_start > 0
                    and self._get_period_key(datetimes[tail_start - 1], period) == latest_key
                ):
                    tail_start -= 1
                tail_count = len(datetimes) - tail_start
                if latest_key is not None and tail_count < period:
                    state["current_key"] = latest_key
                    state["current_count"] = tail_count
                    state["current_open"] = float(ohlcv[0, tail_start])
                    state["current_high"] = float(np.max(ohlcv[1, tail_start:]))
                    state["current_low"] = float(np.min(ohlcv[2, tail_start:]))
                    state["current_close"] = float(ohlcv[3, -1])
                    state["current_volume"] = float(np.sum(ohlcv[4, tail_start:]))

            self._period_states[period] = state

        self.short_direction = self._signal_from_state(self._period_states[self.SHORT_PERIOD])
        self.middle_direction = self._signal_from_state(self._period_states[self.MIDDLE_PERIOD])
        self.long_direction = self._signal_from_state(self._period_states[self.LONG_PERIOD])

        highs = self.am.high_array
        lows = self.am.low_array
        closes = self.am.close_array
        start = max(1, len(closes) - self.ATR_WINDOW)
        for i in range(start, len(closes)):
            previous_close = closes[i - 1]
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - previous_close),
                abs(lows[i] - previous_close),
            )
            self._atr_values.append(float(tr))
        if self._atr_values:
            self.last_atr = float(np.mean(self._atr_values))
        self._previous_close = float(closes[-1])
        self._indicator_inited = True

    def _new_period_state(self) -> dict:
        return {
            "highs": deque(maxlen=self.RSV_WINDOW),
            "lows": deque(maxlen=self.RSV_WINDOW),
            "k": None,
            "d": None,
            "count": 0,
            "signal": 0,
            "current_key": None,
            "current_count": 0,
            "current_open": 0.0,
            "current_high": 0.0,
            "current_low": 0.0,
            "current_close": 0.0,
            "current_volume": 0.0,
        }

    def _advance_kd_state(self, state: dict, high: float, low: float, close: float) -> int:
        state["highs"].append(float(high))
        state["lows"].append(float(low))

        highest = max(state["highs"])
        lowest = min(state["lows"])
        rsv = 50.0 if highest == lowest else (float(close) - lowest) / (highest - lowest) * 100

        if state["k"] is None:
            state["k"] = rsv
            state["d"] = rsv
        else:
            state["k"] = (2.0 / 3.0) * state["k"] + (1.0 / 3.0) * rsv
            state["d"] = (2.0 / 3.0) * state["d"] + (1.0 / 3.0) * state["k"]

        state["count"] += 1
        state["signal"] = self._signal_from_state(state)
        return state["signal"]

    def _signal_from_state(self, state: dict) -> int:
        if state["count"] < self.RSV_WINDOW or state["k"] is None:
            return 0
        if state["k"] >= self.RED_THRESHOLD and state["k"] >= state["d"]:
            return 1
        if state["k"] <= self.GREEN_THRESHOLD and state["k"] <= state["d"]:
            return -1
        return 0

    def _advance_period(self, bar: BarData, period: int):
        state = self._period_states[period]
        if period == 1:
            signal = self._advance_kd_state(
                state, bar.high_price, bar.low_price, bar.close_price
            )
            return signal, True

        key = self._get_period_key(bar.datetime, period)
        if key is None:
            return state["signal"], False

        if state["current_key"] is None:
            self._start_period_bar(state, key, bar)
            return state["signal"], False

        if key == state["current_key"]:
            state["current_count"] += 1
            state["current_high"] = max(state["current_high"], bar.high_price)
            state["current_low"] = min(state["current_low"], bar.low_price)
            state["current_close"] = bar.close_price
            state["current_volume"] += bar.volume
            return state["signal"], False

        updated = False
        if state["current_count"] >= period:
            self._advance_kd_state(
                state,
                state["current_high"],
                state["current_low"],
                state["current_close"],
            )
            updated = True

        self._start_period_bar(state, key, bar)
        return state["signal"], updated

    @staticmethod
    def _start_period_bar(state: dict, key: int, bar: BarData):
        state["current_key"] = key
        state["current_count"] = 1
        state["current_open"] = bar.open_price
        state["current_high"] = bar.high_price
        state["current_low"] = bar.low_price
        state["current_close"] = bar.close_price
        state["current_volume"] = bar.volume

    def _advance_atr(self, bar: BarData):
        previous_close = self._previous_close
        if previous_close is None:
            tr = bar.high_price - bar.low_price
        else:
            tr = max(
                bar.high_price - bar.low_price,
                abs(bar.high_price - previous_close),
                abs(bar.low_price - previous_close),
            )
        self._atr_values.append(float(tr))
        self.last_atr = float(np.mean(self._atr_values))
        self._previous_close = bar.close_price

    # ================================================================
    #  时间与重采样
    # ================================================================

    def _get_day_minutes(self, dt: datetime) -> int:
        """兼容辅助方法: 返回各交易时段内从零开始的分钟数。"""
        session_position = self._get_session_position(dt)
        return -1 if session_position is None else session_position[1]

    def _get_session_position(self, dt: datetime):
        h, m = dt.hour, dt.minute
        if h > 17 or (h == 17 and m >= 15):
            return 0, (h - 17) * 60 + m - 15
        if h < 3:
            return 0, 405 + h * 60 + m
        if (h == 9 and m >= 15) or 10 <= h < 12:
            return 1, (h - 9) * 60 + m - 15
        if 13 <= h < 17:
            return 2, (h - 13) * 60 + m
        return None

    def _get_period_key(self, dt: datetime, period: int):
        session_position = self._get_session_position(dt)
        if session_position is None:
            return None
        session, session_minute = session_position
        trading_day = dt.toordinal() + 1 if dt.hour >= 17 else dt.toordinal()
        return trading_day * 10_000_000 + session * 1_000_000 + session_minute // period

    def _check_period_boundary(self, bar: BarData, period: int) -> bool:
        """检查是否在指定周期K线组完成时"""
        group = self._get_period_key(bar.datetime, period)
        if group is None:
            return False

        if period == self.MIDDLE_PERIOD:
            if group != self._last_4min_group:
                self._last_4min_group = group
                return True
            return False
        elif period == self.LONG_PERIOD:
            if group != self._last_14min_group:
                self._last_14min_group = group
                return True
            return False
        else:
            return True  # 1分钟每根都是边界

    def _resample_by_time(self, ohlcv: np.ndarray, datetimes: list, period: int) -> np.ndarray:
        """S9/S10: 按交易时段重采样, 每个时段独立零点, 丢弃不完整组"""
        if period == 1:
            return ohlcv.copy()

        n = ohlcv.shape[1]
        if n == 0:
            return np.empty((5, 0))

        # 计算每个bar的交易日+时段+组编号
        group_ids = np.empty(n, dtype=np.int64)
        for i, dt in enumerate(datetimes):
            key = self._get_period_key(dt, period)
            # 无效时间各自成组, 防止连续无效bar被误聚合成完整K线。
            group_ids[i] = key if key is not None else -(i + 1)

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
        result[0, :] = ohlcv[0, group_starts]                          # open
        result[1, :] = np.maximum.reduceat(ohlcv[1, :], group_starts)  # high
        result[2, :] = np.minimum.reduceat(ohlcv[2, :], group_starts)  # low
        group_ends = np.append(group_starts[1:], n) - 1
        result[3, :] = ohlcv[3, group_ends]                             # close
        result[4, :] = np.add.reduceat(ohlcv[4, :], group_starts)      # volume

        # S10: 丢弃不完整组 (bar数 < period)
        group_sizes = np.diff(np.append(group_starts, n))
        complete_mask = group_sizes >= period
        result = result[:, complete_mask]

        return result

    # ================================================================
    #  止损止盈预挂单
    # ================================================================

    def _refresh_order_pending(self):
        self.order_pending = bool(self.entry_orderids or self.exit_orderids)

    def _submit_forced_exit(self, bar: BarData, reason: str):
        """撤销开仓/保护委托并提交可追踪的强制退出单。"""
        self.force_exit_reason = reason
        self.protection_sync_pending = False
        self.stop_replace_pending = False

        for vt_orderid in list(self.entry_orderids):
            self.cancel_order(vt_orderid)

        self._cancel_protective_orders()

        # 实盘撤单是异步的。等撤单回报到齐后, 下一根bar再提交退出单。
        if self.entry_orderids or self.stop_orderid or self.tp_orderid or self.exit_orderids:
            self._refresh_order_pending()
            return

        volume = abs(self.pos)
        if volume == 0:
            self.force_exit_reason = ""
            self._refresh_order_pending()
            return

        if self.pos > 0:
            ids = self.sell(bar.close_price - self.FIXED_SLIPPAGE, volume)
        else:
            ids = self.cover(bar.close_price + self.FIXED_SLIPPAGE, volume)

        if ids:
            self.exit_orderids.update(ids)
            self.exit_order_placed_bar = self.bar_count
            self.write_log(
                f"提交强制退出单: reason={reason} pos={self.pos} ids={ids}"
            )
        self._refresh_order_pending()

    def _cancel_protective_orders(self):
        ids = [order_id for order_id in (self.stop_orderid, self.tp_orderid) if order_id]
        self.protection_cancel_ids.update(ids)
        for order_id in ids:
            self.cancel_order(order_id)

    def _request_protection_sync(self):
        """按实际持仓量重建止损/止盈, 用于部分成交和部分平仓。"""
        if self.pos == 0 or self.force_exit_reason:
            return

        desired_volume = abs(self.pos)
        if (
            desired_volume == self.protected_volume
            and self.stop_orderid
            and self.tp_orderid
            and not self.protection_sync_pending
        ):
            return

        self.protection_sync_pending = True
        self.stop_replace_pending = False
        self._cancel_protective_orders()
        self._try_place_protective_orders()

    def _try_place_protective_orders(self):
        if (
            not self.protection_sync_pending
            or self.protection_cancel_ids
            or self.force_exit_reason
            or self.pos == 0
        ):
            return

        vol = abs(self.pos)

        if self.pos > 0:
            if self.stop_price is not None and not self.stop_orderid:
                ids = self.sell(self.stop_price, vol, stop=True)
                if ids:
                    self.stop_orderid = ids[0]
                    self.write_log(f"预挂多头止损单: stop={self.stop_price:.2f} id={ids[0]}")
            if self.take_profit is not None and not self.tp_orderid:
                ids = self.sell(self.take_profit, vol)
                if ids:
                    self.tp_orderid = ids[0]
                    self.write_log(f"预挂多头止盈单: tp={self.take_profit:.2f} id={ids[0]}")
        else:
            if self.stop_price is not None and not self.stop_orderid:
                ids = self.cover(self.stop_price, vol, stop=True)
                if ids:
                    self.stop_orderid = ids[0]
                    self.write_log(f"预挂空头止损单: stop={self.stop_price:.2f} id={ids[0]}")
            if self.take_profit is not None and not self.tp_orderid:
                ids = self.cover(self.take_profit, vol)
                if ids:
                    self.tp_orderid = ids[0]
                    self.write_log(f"预挂空头止盈单: tp={self.take_profit:.2f} id={ids[0]}")

        if self.stop_orderid and self.tp_orderid:
            self.protected_volume = vol
            self.protection_sync_pending = False

    def _try_replace_stop_order(self):
        if (
            not self.stop_replace_pending
            or self.stop_orderid
            or self.protection_cancel_ids
            or self.force_exit_reason
            or self.pos == 0
            or self.stop_price is None
        ):
            return

        volume = abs(self.pos)
        if self.pos > 0:
            ids = self.sell(self.stop_price, volume, stop=True)
        else:
            ids = self.cover(self.stop_price, volume, stop=True)
        if ids:
            self.stop_orderid = ids[0]
            self.stop_replace_pending = False
            self.protected_volume = volume
            self.write_log(f"移动止损单已更新: stop={self.stop_price:.2f} id={ids[0]}")

    def _update_trailing_stop(self, bar: BarData):
        """止损上移时更新停止单 (多头只上移, 空头只下移)"""
        atr = self.last_atr
        last_close = bar.close_price

        if self.pos > 0:
            new_stop = last_close - self.STOP_LOSS_MULTIPLIER * atr * (1 + self.SLIPPAGE_RATIO)
            if self.stop_price is None:
                self.stop_price = new_stop
            elif new_stop > self.stop_price:
                old_stop = self.stop_price
                self.stop_price = new_stop
                self.stop_replace_pending = True
                if self.stop_orderid:
                    self.protection_cancel_ids.add(self.stop_orderid)
                    self.cancel_order(self.stop_orderid)
                self._try_replace_stop_order()
                self.write_log(f"止损上移: {old_stop:.2f}→{new_stop:.2f}")
        elif self.pos < 0:
            new_stop = last_close + self.STOP_LOSS_MULTIPLIER * atr * (1 + self.SLIPPAGE_RATIO)
            if self.stop_price is None:
                self.stop_price = new_stop
            elif new_stop < self.stop_price:
                old_stop = self.stop_price
                self.stop_price = new_stop
                self.stop_replace_pending = True
                if self.stop_orderid:
                    self.protection_cancel_ids.add(self.stop_orderid)
                    self.cancel_order(self.stop_orderid)
                self._try_replace_stop_order()
                self.write_log(f"止损下移: {old_stop:.2f}→{new_stop:.2f}")

    def _cancel_opposite_orders(self, trade: TradeData):
        """平仓后撤销旧保护单, 剩余仓位按实际数量重新保护。"""
        self.protected_volume = 0
        if self.pos == 0:
            self.protection_sync_pending = False
            self.stop_replace_pending = False
            self._cancel_protective_orders()
        elif not self.force_exit_reason:
            self._request_protection_sync()
