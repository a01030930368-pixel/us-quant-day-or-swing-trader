"""규칙서 b01: 첫 3분봉 큰 거래대금 양봉 -> 조정 -> 첫 3분봉 종가 재돌파 (당일 매매).

한 종목-하루의 정규장 1분봉(mi = 9:30부터 몇 분째)을 받아 진입 신호와 청산 결과를 낸다.
신호 판단은 3분봉, 손절·익절 체결은 1분봉으로 본다. 같은 1분 안에 손절가와 익절가가 모두 닿으면 손절로 친다(보수적).
"""
from dataclasses import dataclass, replace

import numpy as np

CLOSE_MIN = 385  # 15:55에 남은 물량 전량 매도


@dataclass(frozen=True)
class Params:
    name: str
    code: str = ""
    base: str = "first"        # "first" = 첫 3분봉만 / "any" = 11:00 전에 처음 기준을 넘은 3분봉 (D6)
    min_dv: float = 5e6        # 기준봉 거래대금 하한 (달러)
    adv_frac: float = 0.05     # 기준봉 거래대금 >= 20일 평균 하루 거래대금의 5%
    vol_filter: bool = False   # B3·B5: 조정 중 거래량 감소, 돌파봉 거래량 증가
    pull_ratio: float = 0.30   # B3: 조정 구간 평균 거래량 <= 기준봉의 30%
    brk_mult: float = 2.0      # B5: 돌파봉 거래량 > 조정 구간 평균의 2배
    entry: str = "next_open"   # "next_open" = 돌파 3분봉 다음 봉 시가 / "touch" = 닿는 순간 (D3)
    deadline: int = 90         # 매수 마감 (9:30부터 분). 90 = 11:00
    tp: float | None = 0.03    # 익절 (+3%). None = 장 마감까지 (D5)
    max_stop: float = 0.02     # 손절: 조정 저점 아래 또는 -2% 중 가까운 쪽
    base_last: int = 90        # D6 기준봉은 11:00 전에 시작한 봉까지


D1 = Params("D1 기본", "D1")
VARIANTS = [
    D1,
    replace(D1, name="D2 거래량 꿀팁", code="D2", vol_filter=True),
    replace(D1, name="D3 닿는 순간 매수", code="D3", entry="touch"),
    replace(D1, name="D4a 익절 +2%", code="D4a", tp=0.02),
    replace(D1, name="D4b 익절 +5%", code="D4b", tp=0.05),
    replace(D1, name="D5 익절 없이 마감까지", code="D5", tp=None),
    replace(D1, name="D6 아무 3분봉 기준", code="D6", base="any"),
    replace(D1, name="민감도 거래대금 300만$", code="S-dv3", min_dv=3e6),
    replace(D1, name="민감도 거래대금 1,000만$", code="S-dv10", min_dv=10e6),
    replace(D1, name="민감도 D2 조정거래량 20%", code="S-pr20", vol_filter=True, pull_ratio=0.20),
    replace(D1, name="민감도 D2 조정거래량 50%", code="S-pr50", vol_filter=True, pull_ratio=0.50),
    replace(D1, name="민감도 매수마감 10:30", code="S-1030", deadline=60),
    replace(D1, name="민감도 매수마감 12:00", code="S-1200", deadline=150),
]


def three_min(mi, o, h, l, c, v):
    """1분봉 -> 3분봉. 빈 봉은 빼고 (k, o, h, l, c, v, dv) 배열을 돌려준다."""
    k = mi // 3
    ks, start = np.unique(k, return_index=True)
    end = np.r_[start[1:], len(k)]
    dv = c * v
    return dict(
        k=ks,
        o=o[start],
        h=np.maximum.reduceat(h, start),
        l=np.minimum.reduceat(l, start),
        c=c[end - 1],
        v=np.add.reduceat(v, start),
        dv=np.add.reduceat(dv, start),
    )


def find_entry(day, b3, adv, p: Params):
    """진입이 있으면 dict(i=1분봉 위치, px=매수가, pull_low, base_k), 없으면 None."""
    thr = max(p.min_dv, p.adv_frac * adv)
    ok = (b3["c"] > b3["o"]) & (b3["dv"] >= thr)
    if p.base == "first":
        if len(b3["k"]) == 0 or b3["k"][0] != 0 or not ok[0]:
            return None
        b = 0
    else:
        idx = np.flatnonzero(ok & (b3["k"] * 3 < p.base_last))
        if len(idx) == 0:
            return None
        b = idx[0]
    bo, bc, bv = b3["o"][b], b3["c"][b], b3["v"][b]
    mi = day["mi"]
    dipped = False
    pull_low = np.inf
    vols = []
    for j in range(b + 1, len(b3["k"])):
        start = b3["k"][j] * 3
        if start >= p.deadline:
            return None
        lo, hi, cl, vo = b3["l"][j], b3["h"][j], b3["c"][j], b3["v"][j]
        if p.entry == "touch" and dipped and hi > bc:
            # 이 3분봉 안에서 처음 첫봉 종가에 닿은 1분을 찾는다
            mins = np.flatnonzero((mi >= start) & (mi < start + 3))
            for t in mins:
                if day["l"][t] <= bo:
                    return None
                if day["h"][t] >= bc:
                    return dict(i=t, px=max(day["o"][t], bc), pull_low=pull_low, base_k=b3["k"][b], base_dv=b3["dv"][b], same_bar=True)
                pull_low = min(pull_low, day["l"][t])
        pull_low = min(pull_low, lo) if dipped or lo < bc else pull_low
        if pull_low <= bo:
            return None  # B2: 조정이 첫봉 시가 아래로 무너짐
        if not dipped:
            if lo < bc:
                dipped = True  # B1: 첫봉 종가 아래로 내려온 적이 있어야 함
            vols.append(vo)
            continue
        if p.entry == "next_open" and cl > bc:
            avg = np.mean(vols) if vols else 0.0
            if not p.vol_filter or (avg <= p.pull_ratio * bv and vo > p.brk_mult * avg):
                t = np.searchsorted(mi, start + 3)
                if t >= len(mi) or mi[t] >= CLOSE_MIN:
                    return None
                return dict(i=t, px=day["o"][t], pull_low=pull_low, base_k=b3["k"][b], base_dv=b3["dv"][b], same_bar=False)
            dipped = False  # 거래량 조건이 안 맞은 돌파는 버리고 다시 조정을 기다림
        vols.append(vo)
    return None


def simulate_exit(day, e, p: Params):
    """매수 후 1분봉으로 손절·익절·마감 청산. dict(gross, exit_mi, reason, stop_dist) 또는 None."""
    px = e["px"]
    stop = max(e["pull_low"] - 0.01, px * (1 - p.max_stop))
    if stop >= px:
        return None
    # 닿는 순간 산 1분봉은 고가·저가의 순서를 모르므로 다음 1분부터 본다
    i = e["i"] + (1 if e.get("same_bar") else 0)
    mi, o, h, l = day["mi"][i:], day["o"][i:], day["h"][i:], day["l"][i:]
    n_close = np.searchsorted(mi, CLOSE_MIN)
    hit_stop = np.flatnonzero(l[:n_close] <= stop)
    hit_tp = np.flatnonzero(h[:n_close] >= px * (1 + p.tp)) if p.tp is not None else np.array([], int)
    s = hit_stop[0] if len(hit_stop) else n_close
    t = hit_tp[0] if len(hit_tp) else n_close
    if s <= t and s < n_close:
        xp, xi, why = min(o[s], stop), s, "손절"
    elif t < n_close:
        xp, xi, why = max(o[t], px * (1 + p.tp)), t, "익절"
    elif n_close < len(mi):
        xp, xi, why = o[n_close], n_close, "마감"
    else:
        xp, xi, why = day["c"][-1], len(mi) - 1, "마감"
    return dict(gross=xp / px - 1, exit_mi=int(mi[xi]), reason=why, stop_dist=1 - stop / px)


def run_day(day, adv, variants=VARIANTS):
    """한 종목-하루에 대해 모든 변형의 거래를 낸다."""
    b3 = three_min(day["mi"], day["o"], day["h"], day["l"], day["c"], day["v"])
    out = []
    for vi, p in enumerate(variants):
        e = find_entry(day, b3, adv, p)
        if e is None:
            continue
        x = simulate_exit(day, e, p)
        if x is None:
            continue
        out.append(dict(variant=vi, entry_mi=int(day["mi"][e["i"]]), entry_px=e["px"], base_k=int(e["base_k"]),
                        base_rel=e["base_dv"] / adv, **x))
    return out
