"""당일 매매 거래 목록 -> 계좌 수익 곡선과 성과 지표.

자금 관리(규칙서 5장): 한 번 손절 시 계좌 0.5% 손실이 되게 비중을 정하되 한 종목 최대 1/3,
동시 보유 최대 3종목, 그날 확정 손실이 계좌 -1.5%에 닿으면 새 진입 중단. 매일 장 마감에 전부 현금.
"""
import numpy as np
import pandas as pd

RISK = 0.005
MAX_W = 1 / 3
MAX_POS = 3
DAY_STOP = -0.015


def net_return(gross, cost):
    """cost = 한쪽(매수 또는 매도) 비용 비율. 수수료 + 미끄러짐."""
    return (1 + gross) * (1 - cost) / (1 + cost) - 1


def daily_returns(trades, days, cost, max_w=MAX_W, max_pos=MAX_POS):
    """trades: DataFrame(d, entry_mi, exit_mi, gross, stop_dist, rank). 하루 수익률 Series와 실제 체결된 거래."""
    t = trades.copy()
    t["net"] = net_return(t.gross, cost)
    t["w"] = np.minimum(RISK / t.stop_dist.clip(lower=1e-4), max_w)
    taken = []
    for d, g in t.sort_values(["d", "entry_mi", "rank"]).groupby("d", sort=False):
        held = []  # (exit_mi, pnl)
        for r in g.itertuples():
            done = sum(p for x, p in held if x <= r.entry_mi)
            if done <= DAY_STOP:
                break
            if sum(1 for x, _ in held if x > r.entry_mi) >= max_pos:
                continue
            held.append((r.exit_mi, r.w * r.net))
            taken.append(r.Index)
    k = t.loc[taken]
    ret = (k.w * k.net).groupby(k.d).sum()
    return ret.reindex(days, fill_value=0.0), k


def stats(ret):
    eq = (1 + ret).cumprod()
    years = len(ret) / 252
    cagr = eq.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    mdd = (eq / eq.cummax() - 1).min()
    sd = ret.std()
    sharpe = ret.mean() / sd * np.sqrt(252) if sd > 0 else np.nan
    return dict(cagr=cagr, mdd=mdd, sharpe=sharpe, total=eq.iloc[-1] - 1)


def yearly(ret):
    return (1 + ret).groupby(pd.DatetimeIndex(ret.index).year).prod() - 1
