"""
vnpy CTA 策略: 周期谐波震荡策略 (Period Harmonic Oscillation) v3.2

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
  S14: 4min/14min信号缓存, 仅在边界时重算(性能优化)
  S15: backtest_hsi.py用Direction/Offset枚举, 修复语法错误
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
    """周期谐波震荡策略 - vnpy CTA版本 v3.2"""

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

        # S14: 信号缓存 (4min/14min只在边界时重算)
        self._cached_4min_signal: int = 0
        self._cached_14min_signal: int = 0

        # 订单ID追踪
        self.stop_orderid: str = ""
        self.tp_orderid: str = ""
        self.entry_orderid: str = ""
        self.order_placed_bar: int = 0
        self.active_order_ids: set = set()

        self._day_just_changed: bool = False

    # ================================================================
    #  vnpy 回调
    # ================================================================

    def on_init(self):
        self.write_log("策略初始化 PeriodHarmonicStrategy v3.2")
        try:
            self.load_bar(10)
        except Exception as e:
            self.write_log(f"load_bar异常(非致命): {e}")

    def on_start(self):
        self.write_log("策略启动 v3.2")
        self.put_event()

    def on_stop(self):
        self.write_log("策略停止 v3.2")
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

        # S12: 开仓单超时检查 (2根K线未成交则撤单)
        if self.order_pending and self.entry_orderid:
            if self.bar_count - self.order_placed_bar >= 2:
                self.write_log(f"开仓单超时撤单: bar_count={self.bar_count} placed={self.order_placed_bar}")
                self.cancel_order(self.entry_orderid)
                self.entry_orderid = ""
                self.order_pending = False

        # 1. 持仓变化检测
        if self.pos != self.last_pos:
            self.order_pending = False
            self.last_pos = self.pos
            self.active_order_ids.clear()

            if self.pos == 0:
                # S3: 仓位归零, 清除止损止盈, 撤销所有残留订单
                self.stop_price = None
                self.take_profit = None
                self.entry_price = None
                self.entry_volume = 0
                self.stop_orderid = ""
                self.tp_orderid = ""
                self.cancel_all()
                self.write_log(f"仓位归零, 清除止损止盈 pos=0 bar_count={self.bar_count}")
            elif self.pos != 0 and self.stop_price is not None:
                # S4: 开仓成交后预挂止损止盈单
                self._place_stop_profit_orders(bar)

        # 2. 交易日切换
        self._check_day_rollover(bar)

        # S6: 日切时清除状态
        if self._day_just_changed:
            if self.long_direction != 0 or self.middle_direction != 0:
                self.write_log(f"日切: long_dir={self.long_direction} mid_dir={self.middle_direction}")
            self.prev_short_direction = 0

        # 3. 连亏暂停触发 (只设倒计时, 不return)
        if self.loss_num >= self.LOSS_CHANCE:
            self.pausing_countdown = self.PAUSING_PERIOD * self.LONG_PERIOD
            self.loss_num = 0
            self.write_log(f"连亏暂停 {self.pausing_countdown} 根K线")

        # 4. 定时平仓 (优先于一切风控, 确保收盘前能平掉)
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
                    sell_price = bar.close_price - self.FIXED_SLIPPAGE
                    self.sell(sell_price, abs(self.pos))
                elif self.pos < 0:
                    buy_price = bar.close_price + self.FIXED_SLIPPAGE
                    self.cover(buy_price, abs(self.pos))
                self.order_pending = True
            self.put_event()
            return

        # 收盘后窗口内禁止开新仓 (但允许已有仓位管理)
        in_close_guard = (
            abs(current_minute - self.CLOSE_TIME_1) <= self.CLOSE_GUARD_WINDOW or
            abs(current_minute - self.CLOSE_TIME_2) <= self.CLOSE_GUARD_WINDOW
        )

        # 5. 信号计算 (分层更新) - 在风控检查之前, 因为14分钟反转需要信号
        if not in_close_guard:
            self._update_signals(bar)

        # 6. 更新止损止盈 (S11: 止盈只在开仓时设置)
        # 止损预挂单模式下, 止损上移时更新停止单
        if self.pos != 0 and self.stop_price is not None:
            self._update_trailing_stop(bar)

        # 7a. 14分钟反转平仓 (优先于暂停/亏损/order_pending, 确保持仓能退出)
        if self.pos > 0 and self.long_direction == -1:
            self.write_log(f"14分钟转空, 平多仓 close={bar.close_price} bar_count={self.bar_count}")
            self.cancel_all()
            self.sell(bar.close_price - self.FIXED_SLIPPAGE, abs(self.pos))
            if self.trading:
                self.order_pending = True
            self.put_event()
            return

        elif self.pos < 0 and self.long_direction == 1:
            self.write_log(f"14分钟转多, 平空仓 close={bar.close_price} bar_count={self.bar_count}")
            self.cancel_all()
            self.cover(bar.close_price + self.FIXED_SLIPPAGE, abs(self.pos))
            if self.trading:
                self.order_pending = True
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
                self.order_pending = True
                self.entry_orderid = ids[0]
                self.order_placed_bar = self.bar_count
                self.active_order_ids.update(ids)

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
                self.order_pending = True
                self.entry_orderid = ids[0]
                self.order_placed_bar = self.bar_count
                self.active_order_ids.update(ids)

        self.put_event()

    def on_order(self, order: OrderData):
        """S2: 处理订单状态变化"""
        # 使用枚举比较
        if order.status == Status.REJECTED:
            self.write_log(f"订单被拒绝: {order.vt_orderid} 价格={order.price}")
            self.order_pending = False
            self.entry_orderid = ""
        elif order.status == Status.CANCELLED:
            self.write_log(f"订单被撤销: {order.vt_orderid}")
            if order.vt_orderid == self.entry_orderid:
                self.order_pending = False
                self.entry_orderid = ""

        # 非活跃订单从追踪集合移除
        if not order.is_active():
            self.active_order_ids.discard(order.vt_orderid)

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
            self.entry_orderid = ""

        elif is_close and self.entry_price is not None and self.prev_direction != 0:
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

        self.put_event()

    def on_stop_order(self, stop_order: StopOrder):
        """S2: 停止单状态变化"""
        # vn.py 4.4 StopOrder 属性是 stop_orderid, 不是 vt_orderid
        if stop_order.status.name == "CANCELLED":
            self.write_log(f"停止单被撤销")
            if stop_order.stop_orderid == self.stop_orderid:
                self.stop_orderid = ""
        elif stop_order.status.name == "TRIGGERED":
            self.write_log(f"停止单已触发, 等待成交")
            if stop_order.stop_orderid == self.stop_orderid:
                self.stop_orderid = ""
            # 止损触发 → 撤销止盈单
            if self.tp_orderid:
                self.cancel_order(self.tp_orderid)
                self.tp_orderid = ""

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
        """分层更新三个周期方向 (S14: 4min/14min仅在边界时重算)"""

        # 1分钟信号每根bar更新 (数据量小, 直接算)
        new_short = self._calc_signal_for_period(self.SHORT_PERIOD)
        self.prev_short_direction = self.short_direction
        self.short_direction = new_short

        # 4分钟信号: 仅在组边界时重算, 否则用缓存
        if self._check_period_boundary(bar, self.MIDDLE_PERIOD):
            new_middle = self._calc_signal_for_period(self.MIDDLE_PERIOD)
            self._cached_4min_signal = new_middle
            if new_middle != self.middle_direction:
                self.write_log(
                    f"4分钟信号更新: {self.middle_direction}→{new_middle} bar_count={self.bar_count}"
                )
            self.middle_direction = new_middle
        else:
            self.middle_direction = self._cached_4min_signal

        # 14分钟信号: 仅在组边界时重算, 否则用缓存
        if self._check_period_boundary(bar, self.LONG_PERIOD):
            new_long = self._calc_signal_for_period(self.LONG_PERIOD)
            self._cached_14min_signal = new_long
            if new_long != self.long_direction:
                self.write_log(
                    f"14分钟信号更新: {self.long_direction}→{new_long} bar_count={self.bar_count}"
                )
            self.long_direction = new_long
        else:
            self.long_direction = self._cached_14min_signal

        # 更新ATR (每根bar更新, 计算量小)
        ohlcv_2d = np.array([
            self.am.open_array,
            self.am.high_array,
            self.am.low_array,
            self.am.close_array,
            self.am.volume_array,
        ]).T
        self.last_atr = self._calculate_atr(ohlcv_2d)

    def _calc_signal_for_period(self, period: int) -> int:
        """计算指定周期的KD信号: +1=多, 0=中性, -1=空"""
        ohlcv = np.array([
            self.am.open_array,
            self.am.high_array,
            self.am.low_array,
            self.am.close_array,
            self.am.volume_array,
        ])  # (5, N)
        datetimes = list(self.bar_datetimes)

        resampled = self._resample_by_time(ohlcv, datetimes, period)
        return self._red_green_signal(resampled)

    # ================================================================
    #  时间与重采样
    # ================================================================

    def _get_day_minutes(self, dt: datetime) -> int:
        """计算交易时段内分钟序号 (按09:15/13:00/17:15分别零点)"""
        h, m = dt.hour, dt.minute

        # 夜盘: 17:15 - 次日02:59 (17:15为零点)
        if h >= 17:
            return (h - 17) * 60 + (m - 15)
        elif h < 5:
            # 次日凌晨, 继续夜盘序号
            # 17:15到23:59 = 404分钟, 00:00继续
            return 404 + h * 60 + m + 1   # 404 = (23-17)*60 + (59-15) + 1
        # 上午盘: 09:15 - 11:59 (09:15为零点)
        elif h >= 9 and h < 12:
            return (h - 9) * 60 + (m - 15)
        # 下午盘: 13:00 - 16:29 (13:00为零点, 加10000与上午区分)
        elif h >= 13 and h < 17:
            return 10000 + (h - 13) * 60 + m
        else:
            # 05:00-08:59 或 12:00-12:59 (午休)
            return -1  # 无效时段

    def _check_period_boundary(self, bar: BarData, period: int) -> bool:
        """检查是否在指定周期K线组完成时"""
        dt = bar.datetime

        # 交易日 (17:00+归次日)
        if dt.hour >= 17:
            tday = dt.toordinal() + 1
        else:
            tday = dt.toordinal()

        day_minutes = self._get_day_minutes(dt)
        if day_minutes < 0:
            return False  # 无效时段

        # 按时段+交易日+组编号生成唯一key
        session = day_minutes // 10000  # 0=夜盘, 0=上午, 1=下午
        session_minutes = day_minutes % 10000 if session >= 1 else day_minutes
        group = tday * 1000000 + session * 100000 + session_minutes // period

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
            if dt.hour >= 17:
                tday = dt.toordinal() + 1
            else:
                tday = dt.toordinal()
            day_min = self._get_day_minutes(dt)
            if day_min < 0:
                # 无效时段: 赋一个唯一的大值, 使其自成一组(之后会被丢弃)
                group_ids[i] = -1
                continue
            session = day_min // 10000
            session_minutes = day_min % 10000 if session >= 1 else day_min
            group_ids[i] = tday * 1000000 + session * 100000 + session_minutes // period

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
        """KD信号: +1=多头(动量延续), 0=中性, -1=空头"""
        if ohlcv.shape[1] < self.RSV_WINDOW:
            return 0
        rsv = self._calculate_rsv(ohlcv[1, :], ohlcv[2, :], ohlcv[3, :])
        k, d = self._calculate_kd(rsv)
        lk, ld = k[-1], d[-1]
        if lk >= self.RED_THRESHOLD and lk >= ld:
            return 1    # 多头
        if lk <= self.GREEN_THRESHOLD and lk <= ld:
            return -1   # 空头
        return 0        # 中性

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
    #  止损止盈预挂单
    # ================================================================

    def _place_stop_profit_orders(self, bar: BarData):
        """S4: 开仓成交后预挂止损止盈单"""
        vol = abs(self.pos)
        if vol == 0:
            return

        if self.pos > 0:
            # 多头: 止损卖出(停止单), 止盈卖出(限价单)
            if self.stop_price is not None and self.stop_orderid == "":
                ids = self.sell(self.stop_price, vol, stop=True)
                if ids:
                    self.stop_orderid = ids[0]
                    self.write_log(f"预挂多头止损单: stop={self.stop_price:.2f} id={ids[0]}")
            if self.take_profit is not None and self.tp_orderid == "":
                ids = self.sell(self.take_profit, vol)
                if ids:
                    self.tp_orderid = ids[0]
                    self.write_log(f"预挂多头止盈单: tp={self.take_profit:.2f} id={ids[0]}")
        elif self.pos < 0:
            # 空头: 止损买入(停止单), 止盈买入(限价单)
            if self.stop_price is not None and self.stop_orderid == "":
                ids = self.cover(self.stop_price, vol, stop=True)
                if ids:
                    self.stop_orderid = ids[0]
                    self.write_log(f"预挂空头止损单: stop={self.stop_price:.2f} id={ids[0]}")
            if self.take_profit is not None and self.tp_orderid == "":
                ids = self.cover(self.take_profit, vol)
                if ids:
                    self.tp_orderid = ids[0]
                    self.write_log(f"预挂空头止盈单: tp={self.take_profit:.2f} id={ids[0]}")

    def _update_trailing_stop(self, bar: BarData):
        """止损上移时更新停止单 (多头只上移, 空头只下移)"""
        atr = self.last_atr
        last_close = bar.close_price

        if self.pos > 0:
            new_stop = last_close - self.STOP_LOSS_MULTIPLIER * atr * (1 + self.SLIPPAGE_RATIO)
            if self.stop_price is None:
                self.stop_price = new_stop
            elif new_stop > self.stop_price:
                # 止损上移: 撤旧单, 挂新单
                old_stop = self.stop_price
                self.stop_price = new_stop
                if self.stop_orderid:
                    self.cancel_order(self.stop_orderid)
                    self.stop_orderid = ""
                vol = abs(self.pos)
                ids = self.sell(new_stop, vol, stop=True)
                if ids:
                    self.stop_orderid = ids[0]
                    self.write_log(f"止损上移: {old_stop:.2f}→{new_stop:.2f} id={ids[0]}")
        elif self.pos < 0:
            new_stop = last_close + self.STOP_LOSS_MULTIPLIER * atr * (1 + self.SLIPPAGE_RATIO)
            if self.stop_price is None:
                self.stop_price = new_stop
            elif new_stop < self.stop_price:
                # 止损下移: 撤旧单, 挂新单
                old_stop = self.stop_price
                self.stop_price = new_stop
                if self.stop_orderid:
                    self.cancel_order(self.stop_orderid)
                    self.stop_orderid = ""
                vol = abs(self.pos)
                ids = self.cover(new_stop, vol, stop=True)
                if ids:
                    self.stop_orderid = ids[0]
                    self.write_log(f"止损下移: {old_stop:.2f}→{new_stop:.2f} id={ids[0]}")

    def _cancel_opposite_orders(self, trade: TradeData):
        """平仓成交时撤销另一侧订单"""
        # 如果是止损单触发的平仓 → 撤止盈
        # 如果是止盈单成交的平仓 → 撤止损
        trade_id = trade.vt_orderid

        if trade_id == self.stop_orderid:
            # 止损成交 → 撤止盈
            if self.tp_orderid:
                self.cancel_order(self.tp_orderid)
                self.tp_orderid = ""
            self.stop_orderid = ""
        elif trade_id == self.tp_orderid:
            # 止盈成交 → 撤止损
            if self.stop_orderid:
                self.cancel_order(self.stop_orderid)
                self.stop_orderid = ""
            self.tp_orderid = ""
        else:
            # 信号平仓或定时平仓 → 撤所有
            if self.stop_orderid:
                self.cancel_order(self.stop_orderid)
                self.stop_orderid = ""
            if self.tp_orderid:
                self.cancel_order(self.tp_orderid)
                self.tp_orderid = ""
