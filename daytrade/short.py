"""규칙서 b26~b30(1~3일 짧은 보유) 계산: 하루 자료 만들기, 지표, 거래 수익.

판단은 15:55 1분봉 종가(반나절 장은 12:55)를 그날 종가 대신 쓰고, 매수·매도는 15:59 1분봉 종가(반나절 12:59)로 한다.
그래서 지표가 그날 종가를 미리 보지 않는다.
합성 3배: 기초 ETF 종가→종가 수익률 × 3을 날마다 곱해 쌓고, 3배 ETF 운용·차입 비용 대신 보유 하루 0.02%를 뺀다.
"""
import numpy as np
import pandas as pd

CLOSE, HALF_CLOSE = 389, 209
DECIDE, HALF_DECIDE = 385, 205
SYN_DAILY_COST = 0.0002


def daily_table(bars, half):
    """bars: 정규장 1분봉 DataFrame(ticker, d, mi, o, h, l, c, v). half: 반나절 장 날짜 집합.
    -> 종목별 하루 표 (ticker, d, o, h, l, c, cd, hd, ld): c = 마감 종가, cd/hd/ld = 판단 시각까지의 종가·고가·저가."""
    b = bars[bars.mi >= 0]
    is_half = b.d.isin(half)
    close_mi = np.where(is_half, HALF_CLOSE, CLOSE)
    decide_mi = np.where(is_half, HALF_DECIDE, DECIDE)
    upto_c = b[b.mi <= close_mi]
    upto_d = b[b.mi <= decide_mi]
    g = upto_c.groupby(["ticker", "d"], sort=False)
    out = g.agg(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"), last_mi=("mi", "max")).reset_index()
    gd = upto_d.groupby(["ticker", "d"], sort=False).agg(cd=("c", "last"), hd=("h", "max"), ld=("l", "min")).reset_index()
    out = out.merge(gd, on=["ticker", "d"], how="left")
    out["half"] = out.d.isin(half)
    # 마감 직전 자료가 없는 날(15:50 전에 끝난 자료)은 마감가를 믿을 수 없어서 뺀다
    need = np.where(out.half, HALF_CLOSE - 10, CLOSE - 10)
    out = out[out.last_mi >= need]
    return out.sort_values(["ticker", "d"]).reset_index(drop=True)


SPLIT_RATIOS = (2, 3, 4, 5, 8, 10, 20)  # 1.5는 3배 ETF의 큰 갭(-33%)과 헷갈려서 뺌


def split_adjust(t):
    """자료 가격이 분할 반영 전이라, 전날 종가 대비 오늘 시가가 분할 비율(±3%)이면 분할로 보고
    그 전 가격을 모두 나눠 맞춘다. f = 그날 가격에 곱할 값. 분할로 본 날 목록도 돌려줌."""
    out, found = [], []
    for tk, g in t.groupby("ticker", sort=False):
        g = g.copy()
        r = (g.o / g.c.shift()).to_numpy()
        step = np.ones(len(g))
        for i in range(1, len(g)):
            for k in SPLIT_RATIOS:
                if abs(r[i] * k - 1) < 0.03:
                    step[i] = k
                elif abs(r[i] / k - 1) < 0.03:
                    step[i] = 1 / k
            if step[i] != 1:
                found.append((tk, g.d.iloc[i], step[i]))
        # i일 전 가격은 i일 이후 모든 분할만큼 나눔
        f = np.r_[np.cumprod(step[::-1])[::-1][1:], 1.0]
        f = 1 / f
        for col in ("o", "h", "l", "c", "cd", "hd", "ld"):
            g[col] = g[col] * f
        g["f"] = f
        out.append(g)
    return pd.concat(out, ignore_index=True), found


def rsi_wilder(close, n=2):
    """윌더 방식 RSI. close: 1차원 배열."""
    c = np.asarray(close, float)
    out = np.full(len(c), np.nan)
    if len(c) <= n:
        return out
    d = np.diff(c)
    up, dn = np.clip(d, 0, None), np.clip(-d, 0, None)
    au, ad = up[:n].mean(), dn[:n].mean()
    for i in range(n, len(c)):
        if i > n:
            au = (au * (n - 1) + up[i - 1]) / n
            ad = (ad * (n - 1) + dn[i - 1]) / n
        out[i] = 100.0 if ad == 0 else 100 - 100 / (1 + au / ad)
    return out


def decision_series(g, n=2):
    """한 종목 하루 표 -> 그날 판단용 지표 (종가 자리에 판단 시각 가격 cd를 넣어 계산).
    rsi2(윌더), sma200·sma5(전날까지 종가 + 오늘 판단가), ibs(판단 시각까지 고가·저가 기준)."""
    c = g.c.to_numpy(float)
    cd = g.cd.to_numpy(float)
    m = len(c)
    up = np.clip(np.diff(c), 0, None)
    dn = np.clip(-np.diff(c), 0, None)
    AU, AD = np.full(m, np.nan), np.full(m, np.nan)  # k일 종가까지의 윌더 평균
    if m > n:
        AU[n], AD[n] = up[:n].mean(), dn[:n].mean()
        for k in range(n + 1, m):
            AU[k] = (AU[k - 1] * (n - 1) + up[k - 1]) / n
            AD[k] = (AD[k - 1] * (n - 1) + dn[k - 1]) / n
    rsi = np.full(m, np.nan)
    x = np.r_[np.nan, cd[1:] - c[:-1]]
    au = (np.r_[np.nan, AU[:-1]] * (n - 1) + np.clip(x, 0, None)) / n
    ad = (np.r_[np.nan, AD[:-1]] * (n - 1) + np.clip(-x, 0, None)) / n
    ok = ~np.isnan(au)
    rsi[ok] = np.where(ad[ok] == 0, 100.0, 100 - 100 / (1 + au[ok] / np.where(ad[ok] == 0, 1, ad[ok])))
    cs = np.r_[0.0, np.cumsum(c)]

    def sma(k):
        out = np.full(m, np.nan)
        i = np.arange(k - 1, m)
        out[i] = (cs[i] - cs[i - k + 1] + cd[i]) / k
        return out

    hd, ld = g.hd.to_numpy(float), g.ld.to_numpy(float)
    rng = hd - ld
    ibs = np.where(rng > 0, (cd - ld) / np.where(rng > 0, rng, 1), np.nan)
    return pd.DataFrame({"rsi2": rsi, "sma200": sma(200), "sma5": sma(5), "ibs": ibs}, index=g.index)


def hold_return(closes, i, j):
    """같은 종목 종가 배열에서 i일 종가 매수 -> j일 종가 매도."""
    return closes[j] / closes[i] - 1


def synthetic_return(closes, i, j, lev=3):
    """기초 ETF 종가로 만든 합성 lev배: i일 종가 매수 -> j일 종가 매도 (하루씩 lev배, 보유 하루 비용 차감)."""
    r = closes[i + 1:j + 1] / closes[i:j] - 1
    return float(np.prod(1 + lev * r) - 1 - SYN_DAILY_COST * (j - i))


def exit_rsi(cd, sma5, i, max_days=10):
    """b29 청산: 매수(i일) 뒤 판단가 > 5일 평균인 첫날 종가, 아니면 max_days일째 종가."""
    n = len(cd)
    for j in range(i + 1, min(i + max_days, n - 1) + 1):
        if cd[j] > sma5[j]:
            return j
    return min(i + max_days, n - 1)


def exit_ibs(ibs, i, hi=0.7, max_days=5):
    n = len(ibs)
    for j in range(i + 1, min(i + max_days, n - 1) + 1):
        if ibs[j] > hi:
            return j
    return min(i + max_days, n - 1)


def exit_rsi_stop(cd, sma5, i, entry, stop, max_days=10):
    """b29 청산 + 손절: 판단가 > 5일 평균이거나 판단가가 매수가보다 stop 넘게 떨어진 첫날 종가, 아니면 max_days일째 종가."""
    n = len(cd)
    for j in range(i + 1, min(i + max_days, n - 1) + 1):
        if cd[j] > sma5[j] or cd[j] <= entry * (1 - stop):
            return j
    return min(i + max_days, n - 1)


def combine_daily(trades, days, cost, idle=None):
    """여러 신호 거래를 한 계좌로 합침.
    trades: [(종목, 날짜 목록, 날마다 수익)] - 날짜 d에 그 거래를 들고 있으면 그날 수익이 path 값.
    그날 들고 있는 거래가 여럿이면 돈을 똑같이 나눔. 아무 거래도 없는 날은 idle(날짜 -> SPY 수익) 보유, None이면 현금.
    비용: 종목별 비중이 바뀐 만큼 × cost(한쪽). 비중은 전날 종가에 정해짐."""
    days = list(days)
    pos = {d: k for k, d in enumerate(days)}
    active = [[] for _ in days]
    for tk, ds, path in trades:
        for d, r in zip(ds, path):
            k = pos.get(d)
            if k is not None:
                active[k].append((tk, r))
    ret = np.zeros(len(days))
    prev_w = {}
    for k, d in enumerate(days):
        w, r = {}, 0.0
        if active[k]:
            share = 1.0 / len(active[k])
            for tk, rr in active[k]:
                w[tk] = w.get(tk, 0.0) + share
                r += share * rr
        elif idle is not None:
            w["SPY"] = 1.0
            r = idle.get(d, 0.0)
        turn = sum(abs(w.get(t, 0.0) - prev_w.get(t, 0.0)) for t in set(w) | set(prev_w))
        ret[k] = r - cost * turn
        prev_w = w
    return pd.Series(ret, index=pd.to_datetime(days))
