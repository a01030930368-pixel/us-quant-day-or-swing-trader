"""부자회사원 CANSLIM 'market turn + box breakout' from a Korean YouTube video, rulebook 11.

When the market turns up after a fall, buy at the close a stock that held a
tight box while the market fell and now breaks the box top on big volume with
a gap or a long white candle, its whole industry rising with it, and that
passes the CANSLIM checks we can get from data (latest quarter EPS and sales
+25%, return on equity 17%+, cash flow 1.2x earnings, top 20% relative
strength). Sell at +20% or -7% (-5% when VIX is above 25). 20% of the account
a stock, at most 5.

The market turn comes from the channel's own 5-day-line rule on the index (the
video found it on KOSPI peaks; we use SPY): a close below the 5-day line that
gaps down the next day, or stays below for 3 days, turns the market down; a
close back above the line that holds for 3 days with at least 2 white candles
turns it up. O'Neil's follow-through and distribution days are the R2 variant.

All numbers are from /mnt/project-files/rulebooks/11-캔슬림-시장반전-박스권돌파-5일선시장판단.md
([원전] said in the video, [해석] our numbers, [근사] data stand-in, [추가] added),
fixed before the first run. SPY volume stands in for index volume [근사].

Not automated [미구현]: institutional buying (I), the 'new product' part of N,
next year's estimates (A), float size (S), the one-off earnings check (C).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import sma
from .strategies import Strategy

MIN_DOLLAR_VOL = 20e6       # [해석]
MIN_PRICE = 10.0            # [해석]
MEMBERS = (500, 400)        # S&P 500 + 400 [해석]
WINDOW = 20                 # B1 trading days after the turn [해석]
BOX_DAYS, BOX = 40, 0.25    # B2 [해석]
BIG_UP, TOP = 0.04, 0.70    # B3 long white candle: +4% and close in the top 30% of the range [해석]
VOL_X = 2.0                 # B4 [해석 '월등히']; R6 1.4 [원전 찰스]
SECTOR = 0.015              # B5 [해석]
HIGH_DAYS, NEAR_HIGH = 252, 0.95    # N1 [해석]
RS_DAYS, RS_TOP = 126, 0.80         # L1 [해석]
W, MAX_POS = 0.20, 5        # B8 [해석]
TP, SL, SL_VIX, VIX_HI = 0.20, 0.07, 0.05, 25.0     # X1 [원전 20%, 7%, 5~6%], VIX 25 [해석]
HALF_TP = 0.21              # X2 [원전 채널 21%]
NEAR50 = 0.02               # X2/X5 re-buy near the 50-day line [해석]
BREAK_VOL, BREAK_DROP = 1.5, -0.03  # X3/X4 [해석]
REENTRY_DAYS = 120          # X3/X5 how long a sold stock is watched for a re-buy [해석]
LONG_HOLD = 252             # R9 about a year [원전 짐 타우브]
X7_PART = 0.25              # [원전 오닐 25%]
# market signals
UP_DAYS, UP_WHITE = 3, 2    # K5 [해석 '양봉 3개 정도']
DOWN_DAYS = 3               # K3 [화면 51:32 '3~4일']
BOTTOM_LOOK, BOTTOM_VOL = 10, 1.5   # K7 [해석]
CHOP_LOOK, CHOP_CROSS, CHOP_NEAR = 10, 4, 0.03     # K8 [해석]
FTD_DAYS, FTD_GAIN = (4, 7), 0.012  # M2 follow-through [원전 채널 설명 / 오닐 표준]
DIST_DROP, DIST_N, DIST_LOOK = -0.002, 5, 25        # M2 distribution [해석: 오닐 표준]

_SHARED = None


def market_5day(o, h, l, c, v, quick: bool = False):
    """The channel's 5-day-line market state on index bars (numpy arrays).

    Returns (up, turns, downs): up[i] is the state after day i's close, turns
    and downs the days the market turned up / down. quick=True is M1d (one
    close above the 5-day line turns it up).
    """
    n = len(c)
    s5 = pd.Series(c).rolling(5, min_periods=5).mean().to_numpy()
    up = np.zeros(n, bool)
    turns, downs = [], []
    state, broke, watch = False, None, None
    for i in range(5, n):
        if state:
            if broke is None:
                if c[i] < s5[i]:
                    broke = i                                        # K2: close below the 5-day line
            elif c[i] >= s5[i]:
                broke = None                                         # K3: won back in time
            elif (i == broke + 1 and o[i] < c[broke]) or i - broke >= DOWN_DAYS:
                state, broke = False, None                           # K2 gap down / K3 not back in 3 days
                downs.append(i)
        else:
            if c[i] < s5[i]:
                watch = None
            elif watch is None and c[i] > s5[i]:
                watch = i                                            # K4b: back above the line
            if watch is not None and (quick or i - watch + 1 >= UP_DAYS):
                white = int(np.sum(c[watch:i + 1] > o[watch:i + 1]))
                if quick or white >= UP_WHITE:                       # K5
                    state, watch = True, None
                    turns.append(i)
                else:
                    watch = None
        up[i] = state
    return up, turns, downs


def market_oneil(o, h, l, c, v):
    """O'Neil: a follow-through day turns the market up, 5 distribution days in 25 turn it down."""
    n = len(c)
    up = np.zeros(n, bool)
    turns, downs = [], []
    state, dist, low, attempt = False, [], c[0], None
    for i in range(1, n):
        r = c[i] / c[i - 1] - 1
        if state:
            if r <= DIST_DROP and v[i] > v[i - 1]:
                dist.append(i)
            dist = [d for d in dist if i - d < DIST_LOOK]
            if len(dist) >= DIST_N:
                state, dist, low, attempt = False, [], c[i], None
                downs.append(i)
        else:
            if c[i] < low:
                low, attempt = c[i], None                            # new low: the rally attempt starts over
            elif attempt is None and r > 0:
                attempt = i                                          # rally day 1
            elif attempt is not None:
                day = i - attempt + 1
                if FTD_DAYS[0] <= day <= FTD_DAYS[1] and r >= FTD_GAIN and v[i] > v[i - 1]:
                    state, dist, attempt = True, [], None
                    turns.append(i)
                elif day > FTD_DAYS[1]:
                    attempt = i if r > 0 else None
        up[i] = state
    return up, turns, downs


def quality_flags(facts: dict[str, dict], index: pd.DatetimeIndex, roe_min: float = 0.17,
                  cash_x: float = 1.2, stale_days: int = 400) -> tuple[pd.DataFrame, int]:
    """A2 from the latest annual report (10-K): net income / equity >= 17% and
    operating cash flow >= 1.2 x net income, usable from the day after filing."""
    def annual(f, tags, instant=False):
        gaap = f.get("facts", {}).get("us-gaap", {})
        for tag in tags:
            rows = [it for it in gaap.get(tag, {}).get("units", {}).get("USD", [])
                    if it.get("form", "").startswith("10-K") and (instant or "start" in it)]
            if not rows:
                continue
            df = pd.DataFrame(rows)
            for col in ("end", "filed") + (() if instant else ("start",)):
                df[col] = pd.to_datetime(df[col])
            if not instant:
                df = df[(df["end"] - df["start"]).dt.days.between(350, 380)]
            return df.sort_values("filed").drop_duplicates("end", keep="first")[["end", "filed", "val"]]
        return pd.DataFrame(columns=["end", "filed", "val"])

    cols, covered = {}, 0
    for t, f in facts.items():
        ni = annual(f, ["NetIncomeLoss", "ProfitLoss"])
        eq = annual(f, ["StockholdersEquity",
                        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"], instant=True)
        cf = annual(f, ["NetCashProvidedByUsedInOperatingActivities",
                        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"])
        out = pd.Series(False, index=index)
        if len(ni) and len(eq) and len(cf):
            m = ni.merge(eq, on="end", suffixes=("", "_eq")).merge(cf, on="end", suffixes=("", "_cf"))
            if len(m):
                covered += 1
                m["filed"] = m[["filed", "filed_eq", "filed_cf"]].max(axis=1)
                m["ok"] = (m["val_eq"] > 0) & (m["val"] / m["val_eq"] >= roe_min) & (m["val_cf"] >= cash_x * m["val"])
                m = m.sort_values("filed").drop_duplicates("filed", keep="last")
                pos = index.searchsorted(m["filed"].values, side="right")
                val = pd.Series(np.nan, index=index)
                known = pd.Series(pd.NaT, index=index)
                for p, row in zip(pos, m.itertuples()):
                    if p < len(index):
                        val.iloc[p] = float(row.ok)
                        known.iloc[p] = row.filed
                val, known = val.ffill(), known.ffill()
                out = (val == 1.0) & ((index.to_series() - known).dt.days <= stale_days)
        cols[t] = out
    return pd.DataFrame(cols, index=index), covered


class Canslim(Strategy):
    """R1: the video's way."""
    name = "canslim_r1"
    label = "R1 캔슬림 시장 반전 박스권 돌파"
    description = ""
    max_positions = MAX_POS
    close_fill = True
    market_rule = "m1"          # "m1", "m1d", "m2", "" (R3)
    bottom = False              # R10 K7
    chop = False                # R11 K8 (+ X7)
    canslim = True              # R4: False (EPS/sales/ROE/cash flow)
    rs = True                   # L1
    sector = True               # R5: False
    new_high = True             # R12: False
    vol_x = VOL_X
    box, box_days, window = BOX, BOX_DAYS, WINDOW
    tp, sl = TP, SL
    exits = "x1"                # "x1", "x2" (R7), "x5" (R8), "x4" (R9)

    def prepare(self, d: dict[str, pd.DataFrame], market: str = "SPY") -> "Canslim":
        self.info = d.get("INFO")
        self.earn = d.get("EarnOK")
        self.qual = d.get("QualOK")
        self.vix = d.get("VIX")
        return super().prepare(d, market)

    # ------------------------------------------------------------------ setup
    def _base(self) -> dict:
        """Arrays shared by every variant of one data set."""
        global _SHARED
        key = (self.c, self.info, self.earn, self.qual, self.vix)
        if _SHARED is not None and all(a is b for a, b in zip(_SHARED[0], key)):
            return _SHARED[1]
        t = self.tradable
        c, o, h, l, v = (x[t] for x in (self.c, self.o, self.h, self.l, self.v))
        info = self.info if self.info is not None else pd.DataFrame(columns=["index", "sub_industry"])
        known = info.reindex(t)
        member = known["index"].isin(MEMBERS).to_numpy()
        value = c * v
        liquid = ((value.rolling(20, min_periods=20).mean() >= MIN_DOLLAR_VOL) & (c >= MIN_PRICE)).to_numpy() & member[None, :]
        C = c.to_numpy(float)
        G = np.full_like(C, np.nan)
        G[1:] = C[1:] / C[:-1] - 1
        # B5: the average move of the other stocks in the same sub-industry today
        sub = known["sub_industry"].fillna("").astype(str).to_numpy()
        peer = np.full_like(C, np.nan)
        Gz = np.where(np.isfinite(G), G, 0.0)
        Gn = np.isfinite(G).astype(float)
        for name in set(sub) - {""}:
            ks = np.nonzero(sub == name)[0]
            if len(ks) < 2:
                continue
            tot, cnt = Gz[:, ks].sum(1, keepdims=True), Gn[:, ks].sum(1, keepdims=True)
            others = np.maximum(cnt - Gn[:, ks], 1)
            peer[:, ks] = np.where(cnt - Gn[:, ks] >= 1, (tot - Gz[:, ks]) / others, np.nan)
        # L1: 126-day return rank among the liquid members
        r126 = (c / c.shift(RS_DAYS) - 1).where(pd.DataFrame(liquid, index=c.index, columns=t))
        rank = r126.rank(axis=1, pct=True).to_numpy(float)
        m = self.c[self.market]
        mk = {x: y[self.market].to_numpy(float) for x, y in
              (("o", self.o), ("h", self.h), ("l", self.l), ("c", self.c), ("v", self.v))}
        b = {"C": C, "O": o.to_numpy(float), "H": h.to_numpy(float), "L": l.to_numpy(float),
             "V": v.to_numpy(float), "G": G, "liquid": liquid, "peer": peer, "rank": rank,
             "avg50": v.rolling(50, min_periods=50).mean().shift(1).to_numpy(float),
             "sma50": sma(c, 50).to_numpy(float),
             "hi252": h.rolling(HIGH_DAYS, min_periods=HIGH_DAYS).max().shift(1).to_numpy(float),
             "mkt": mk, "mret": m.to_numpy(float),
             "markets": {"m1": market_5day(*mk.values()), "m1d": market_5day(*mk.values(), quick=True),
                         "m2": market_oneil(*mk.values())}}
        ok = np.ones_like(C, bool)
        if self.earn is not None:
            ok &= self.earn.reindex(index=c.index, columns=t).fillna(False).astype(bool).to_numpy()
        if self.qual is not None:
            ok &= self.qual.reindex(index=c.index, columns=t).fillna(False).astype(bool).to_numpy()
        b["fund"] = ok
        b["vix"] = (self.vix.reindex(c.index).ffill().to_numpy(float) if self.vix is not None
                    else np.full(len(C), np.nan))
        _SHARED = (key, b)
        return b

    def _boxes(self, b: dict, days: int) -> tuple[np.ndarray, np.ndarray]:
        key = ("box", days)
        if key not in b:
            hi = pd.DataFrame(b["H"]).rolling(days, min_periods=days).max().shift(1).to_numpy()
            lo = pd.DataFrame(b["L"]).rolling(days, min_periods=days).min().shift(1).to_numpy()
            b[key] = (hi, lo)
        return b[key]

    def _prepare(self) -> None:
        b = self._base()
        self.b = b
        self.col = {tk: k for k, tk in enumerate(self.tradable)}
        C, O, H, L, V, G = b["C"], b["O"], b["H"], b["L"], b["V"], b["G"]
        n = len(C)
        # market state and buy window
        if self.market_rule:
            up, turns, downs = b["markets"][self.market_rule]
        else:
            up, turns, downs = np.ones(n, bool), [], []
        self.up, self.downs = up, set(downs)
        mo, mh, ml, mc, mv = (b["mkt"][x] for x in "ohlcv")
        if self.bottom:                                              # K7: a heavy red index day before the turn
            avg20 = pd.Series(mv).rolling(20, min_periods=20).mean().shift(1).to_numpy()
            heavy = (mc < mo) & (mv >= BOTTOM_VOL * avg20)
            turns = [i for i in turns if heavy[max(0, i - UP_DAYS + 1 - BOTTOM_LOOK):i - UP_DAYS + 1].any()]
        window = np.zeros(n, bool)
        if self.market_rule:
            for i in turns:
                window[i:i + self.window] = True
            window &= up
        else:
            window[:] = True
        if self.chop:                                                # K8: choppy near the high -> no new buys
            s5 = pd.Series(mc).rolling(5, min_periods=5).mean().to_numpy()
            side = np.sign(mc - s5)
            cross = np.r_[0, (side[1:] != side[:-1]) & (side[1:] != 0)].astype(float)
            crosses = pd.Series(cross).rolling(CHOP_LOOK, min_periods=1).sum().to_numpy()
            near = mc >= pd.Series(mc).rolling(HIGH_DAYS, min_periods=1).max().to_numpy() * (1 - CHOP_NEAR)
            window &= ~((crosses >= CHOP_CROSS) & near)
        self.window_ok = window
        # stock signal
        hi, lo = self._boxes(b, self.box_days)
        mret = b["mret"]
        k_back = self.box_days
        stock_ret = np.full_like(C, np.nan)
        stock_ret[k_back + 1:] = C[k_back:-1] / C[:-k_back - 1] - 1
        mkt_ret = np.full(n, np.nan)
        mkt_ret[k_back + 1:] = mret[k_back:-1] / mret[:-k_back - 1] - 1
        prevH = np.vstack([np.full((1, C.shape[1]), np.nan), H[:-1]])
        long_white = (G >= BIG_UP) & ((C - L) >= TOP * (H - L))
        sig = (b["liquid"] & (hi / lo - 1 <= self.box) & (stock_ret > mkt_ret[:, None])
               & (C > hi) & ((O >= prevH) | long_white) & (V >= self.vol_x * b["avg50"]))
        if self.sector:
            sig &= b["peer"] >= SECTOR
        if self.canslim:
            sig &= b["fund"]
        if self.rs:
            sig &= b["rank"] >= RS_TOP
        if self.new_high:
            sig &= C >= NEAR_HIGH * b["hi252"]
        self.sig = sig & window[:, None]
        self.begin()

    def begin(self) -> None:
        self.sold: dict[int, int] = {}          # column -> day sold by X4/X5 (for X3/X5 re-buys)
        self.cut: set = set()

    # ------------------------------------------------------------------ entries
    def candidates(self, i: int) -> list[str]:
        ks = list(np.nonzero(self.sig[i])[0])
        ks.sort(key=lambda k: -self.b["G"][i, k])
        out = [self.tradable[k] for k in ks]
        if self.exits in ("x4", "x5") and self.up[i]:
            C, L, V = self.b["C"], self.b["L"], self.b["V"]
            m50, avg = self.b["sma50"], self.b["avg50"]
            for k, day in list(self.sold.items()):
                if i - day > REENTRY_DAYS:
                    self.sold.pop(k)
                    continue
                if self.exits == "x4":                               # X3: dipped under the 50-day line, won it back on volume
                    ok = L[i, k] < m50[i, k] <= C[i, k] and V[i, k] >= BREAK_VOL * avg[i, k]
                else:                                                # X5: market up again, close near the 50-day line
                    ok = abs(C[i, k] / m50[i, k] - 1) <= NEAR50
                if ok and self.tradable[k] not in out:
                    out.append(self.tradable[k])
        return out

    def weight(self, t: str, i: int) -> float:
        return W

    def entry_fill(self, t: str, i: int, open_: float, high: float) -> float | None:
        return self.c[t].iloc[i - 1]                                 # B7: the signal day's close

    def accept_entry(self, t: str, i: int, price: float) -> bool:
        self.sold.pop(self.col[t], None)
        return True

    # ------------------------------------------------------------------ exits
    def initial_stop(self, t: str, i: int, price: float) -> float | None:
        if self.exits == "x4":
            return None
        vix = self.b["vix"][i - 1]
        sl = SL_VIX if self.sl == SL and np.isfinite(vix) and vix > VIX_HI else self.sl
        return price * (1 - sl)

    def take_profit(self, pos: dict, i: int):
        if self.exits in ("x1", "x5"):
            return pos["entry_price"] * (1 + self.tp), 1.0, f"익절 +{self.tp * 100:.0f}%"
        if self.exits == "x2" and not pos.get("halved"):
            target = pos["entry_price"] * (1 + HALF_TP)
            if self.b["H"][i, self.col[pos["ticker"]]] >= target:   # the engine sells today: mark it
                pos["halved"] = pos["x4"] = True
                pos["stop"] = None                                   # the rest follows the 50-day rule
            return target, 0.5, f"익절 +{HALF_TP * 100:.0f}% (절반)"
        return None

    def update_stop(self, pos: dict, i: int) -> float | None:
        return pos.get("stop")

    def close_add(self, pos: dict, i: int):
        if self.exits != "x2" or not pos.get("halved"):
            return None
        k = self.col[pos["ticker"]]
        if self.up[i] and abs(self.b["C"][i, k] / self.b["sma50"][i, k] - 1) <= NEAR50 and i > pos["entry_i"]:
            pos["halved"] = False                                    # X2: refill at the 50-day line, then +21% again
            return W * 0.5
        return None

    def exit_reason(self, pos: dict, i: int):
        k = self.col[pos["ticker"]]
        b = self.b
        if self.exits == "x4" or pos.get("x4"):
            if b["C"][i, k] < b["sma50"][i, k] and b["V"][i, k] >= BREAK_VOL * b["avg50"][i, k] \
                    and b["G"][i, k] <= BREAK_DROP:
                if self.exits == "x4":
                    self.sold[k] = i
                return "대량 거래 50일선 이탈"                               # X4
        if self.exits == "x4" and i - pos["entry_i"] + 2 >= LONG_HOLD:
            return f"{LONG_HOLD}거래일 보유 완료"
        if i in self.downs:
            if self.exits == "x5":
                self.sold[k] = i
                return "시장 하락 전환"                                    # X5
            if self.chop:
                return "시장 하락 전환 (25%)", X7_PART                        # X7
        return None


def _make(name: str, label: str, description: str, **kw) -> type:
    return type(name, (Canslim,), {"name": name.lower(), "label": label, "description": description, **kw})


R1 = _make("Canslim_R1", "R1 캔슬림 시장 반전 박스권 돌파 (지수 5일선)",
           "SPY가 5일선 아래로 닫힌 다음 날 갭 하락하거나 3일 안에 회복 못 하면 하락장, 하락장에서 5일선 위로 닫힌 뒤 "
           "3일 연속 5일선 위·양봉 2개 이상이면 상승 전환. 상승 전환 뒤 20거래일 안(상승장일 때)에, 직전 40일 범위 25% 안에서 "
           "그 40일 동안 SPY보다 강했던 종목이 40일 최고가를 넘어 종가 마감, 갭(시가 ≥ 전날 고가) 또는 +4%·범위 상위 30% "
           "장대양봉, 거래량 50일 평균 2배, 같은 세부 업종 다른 종목 평균 +1.5%, 52주 최고가 95% 이상, 최근 분기 EPS·매출 "
           "+25%, 연간 ROE 17%·영업현금흐름 ≥ 순이익 1.2배, 126일 수익률 상위 20%이면 종가에 20%(최대 5종목). "
           "+20% 익절, -7% 손절(VIX 25 넘으면 -5%).")
R2 = _make("Canslim_R2", "R2 오닐 팔로스루·디스트리뷰션 시장 판단",
           "R1과 같고, 시장 판단을 오닐 방식으로 합니다: 저점 뒤 반등 4~7일째 +1.2%·거래량 증가 = 상승 전환, "
           "25일 안 -0.2% 이상 하락·거래량 증가 5번 = 하락장.", market_rule="m2")
R3 = _make("Canslim_R3", "R3 시장 판단 없음", "R1과 같고, 시장 상승 전환을 기다리지 않습니다.", market_rule="")
R4 = _make("Canslim_R4", "R4 실적 조건 없음 (차트 + 상대 강도만)",
           "R1과 같고, EPS·매출·ROE·현금흐름 조건을 뺍니다(상대 강도 상위 20%는 남김).", canslim=False)
R5 = _make("Canslim_R5", "R5 업종 동반 상승 조건 없음", "R1과 같고, 같은 업종 평균 +1.5% 조건을 뺍니다.", sector=False)
R6 = _make("Canslim_R6", "R6 거래량 1.4배 (찰스 해리스)", "R1과 같고, 거래량을 50일 평균 1.4배로 낮춥니다.", vol_x=1.4)
R7 = _make("Canslim_R7", "R7 +21% 절반 익절 + 50일선 재매수",
           "R1과 같고, +21%에 절반을 팔고(그 뒤 손절 없음) 나머지는 대량 거래 50일선 이탈(거래량 1.5배·-3%)에 팝니다. "
           "상승장에서 종가가 50일선 ±2% 안이면 판 만큼 종가에 다시 채우고 또 +21%를 노립니다.", exits="x2")
R8 = _make("Canslim_R8", "R8 시장 하락 전환에 전량 매도 + 재진입 (찰스)",
           "R1과 같고, 시장 하락 전환 다음 날 시가에 모두 팝니다. 그렇게 판 종목은 120거래일 안에 시장이 상승장이고 "
           "종가가 50일선 ±2% 안이면 종가에 다시 삽니다.", exits="x5")
R9 = _make("Canslim_R9", "R9 50일선 원칙 1년 보유 (짐 타우브, 기록용)",
           "R1과 같이 사고, -7% 손절·+20% 익절 없이 대량 거래 50일선 이탈(거래량 1.5배·-3%)에만 팝니다(최대 252거래일). "
           "판 뒤 120거래일 안에 저가가 50일선 아래·종가는 위·거래량 1.5배인 날이면 종가에 다시 삽니다.", exits="x4")
R10 = _make("Canslim_R10", "R10 + 바닥 대량 음봉 확인",
            "R1과 같고, 상승 전환 직전 10거래일 안에 SPY 거래량 20일 평균 1.5배 음봉이 있었던 전환만 씁니다.", bottom=True)
R11 = _make("Canslim_R11", "R11 + 고점 널뛰기 매수 금지 + 하락 전환 때 25% 매도",
            "R1과 같고, 최근 10거래일에 SPY가 5일선을 4번 이상 오가고 1년 최고가 3% 안이면 새로 사지 않으며, "
            "시장 하락 전환 다음 날 시가에 각 종목 25%를 팝니다.", chop=True)
R12 = _make("Canslim_R12", "R12 신고가 조건 없음", "R1과 같고, 52주 최고가 95% 조건을 뺍니다.", new_high=False)
R13 = _make("Canslim_R13", "R13 5일선 위 하루면 상승 전환 (M1d)",
            "R1과 같고, 하락장에서 SPY가 5일선 위로 한 번 닫히면 바로 상승 전환으로 봅니다.", market_rule="m1d")


def _sweep(base: type, text: str, **kw) -> type:
    tag = "_".join(f"{k}{str(v).replace('.', '')}" for k, v in kw.items())
    return type(f"{base.__name__}_{tag}", (base,), {
        "name": f"{base.name}_{tag}", "label": f"{base.label.split()[0]} 민감도: {text}",
        "description": f"{base.label.split()[0]}과 같고 {text}만 바꿉니다.", **kw})


SWEEP = [_sweep(R1, "박스 폭 15%", box=0.15), _sweep(R1, "박스 폭 35%", box=0.35),
         _sweep(R1, "박스 기간 20일", box_days=20), _sweep(R1, "박스 기간 60일", box_days=60),
         _sweep(R1, "거래량 3배", vol_x=3.0),
         _sweep(R1, "전환 뒤 10일", window=10), _sweep(R1, "전환 뒤 40일", window=40),
         _sweep(R1, "익절 25% 손절 8%", tp=0.25, sl=0.08),
         _sweep(R1, "상대 강도 조건 없음", rs=False),
         _sweep(R4, "시장 판단 없음", market_rule=""),
         _sweep(R4, "시장 판단 없음·업종 조건 없음", market_rule="", sector=False)]
CANSLIM = [R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, *SWEEP]
