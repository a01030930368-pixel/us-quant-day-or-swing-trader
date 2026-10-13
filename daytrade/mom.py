"""규칙서 b10~b15(급등주·갭 종목 장중 매매)의 신호와 체결.

day = 그날 정규장 1분봉 dict(mi, o, h, l, c, v). mi = 9:30부터 몇 분째. 공통 체결 규칙:
- 신호가 1분봉 종가로 확정되면 다음 1분봉 시가에 매수 ("next").
- "넘는 순간" 매수는 기준가 + 0.01달러에 체결, 그 1분봉 시가가 이미 위면 시가 ("stop").
- 손절은 1분봉 저가가 닿으면 손절가(시가가 더 낮으면 시가). 같은 1분봉에서 익절·손절 둘 다 닿으면 손절.
- 15:55 청산 = 15:54 1분봉 종가.
"""
import numpy as np

TICK = 0.01
EOD = 384  # 15:54 1분봉


def vwap(day):
    return np.cumsum(day["c"] * day["v"]) / np.maximum(np.cumsum(day["v"]), 1e-9)


def ema(x, n):
    a, out, e = 2 / (n + 1), np.empty(len(x)), x[0]
    for i, v in enumerate(x):
        e = v if i == 0 else a * v + (1 - a) * e
        out[i] = e
    return out


def ema5_at_minute(day, n=9):
    """5분봉 종가의 n EMA를, 각 1분봉 시점에 '마지막으로 끝난 5분봉' 값으로 (끝난 5분봉이 없으면 nan)."""
    mi, c = day["mi"], day["c"]
    k = mi // 5
    ks, first = np.unique(k, return_index=True)
    last = np.r_[first[1:], len(mi)] - 1
    e = ema(c[last], n)
    out = np.full(len(mi), np.nan)
    for n_, (kk, j) in enumerate(zip(ks, last)):
        out[j:] = e[n_]  # 5분봉이 끝난 마지막 1분봉부터 그 값
    return out, ks, last, e


def relvol_curve(prof):
    """prof: DataFrame(t, f). 시각 m(분)에 '그날 거래량 중 그때까지 나온 비율' 함수."""
    g = prof.groupby("t").f.median().sort_index()
    t = np.r_[0, g.index.to_numpy()]
    f = np.r_[0, g.to_numpy()]
    return lambda m: np.interp(m, t, f)


def cumvol_before(day, j):
    return float(day["v"][:j].sum())


def simulate(day, e, px, stop, target=None, ema5_exit=False, ema1_exit3=False, time_stop=None, until=EOD,
             entry_in_bar=False):
    """e번 1분봉에서 px에 산 뒤 청산. entry_in_bar=True면 그 봉 안에서 산 것(넘는 순간 매수)이라 그 봉 저가로 손절을 본다."""
    mi, o, h, l, c = day["mi"], day["o"], day["h"], day["l"], day["c"]
    j_end = int(np.searchsorted(mi, until, side="right")) - 1
    if time_stop is not None:
        j_end = min(j_end, int(np.searchsorted(mi, mi[e] + time_stop, side="right")) - 1)
    if j_end < e:
        return None
    e5 = ema5_at_minute(day)[0] if ema5_exit else None
    e1 = ema(c, 9) if ema1_exit3 else None
    below = 0
    for j in range(e, j_end + 1):
        first = j == e
        if stop is not None and l[j] <= stop:
            out = stop if first else min(o[j], stop)
            return dict(exit_mi=int(mi[j]), exit_px=out, gross=out / px - 1, reason="손절")
        if target is not None and h[j] >= target:
            out = target if first else max(o[j], target)
            return dict(exit_mi=int(mi[j]), exit_px=out, gross=out / px - 1, reason="익절")
        if j < j_end:
            if ema5_exit and (mi[j] + 1) % 5 == 0 and not np.isnan(e5[j]) and c[j] < e5[j] and mi[j] >= mi[e] + 4:
                return dict(exit_mi=int(mi[j + 1]), exit_px=o[j + 1], gross=o[j + 1] / px - 1, reason="9EMA 이탈")
            if ema1_exit3:
                below = below + 1 if c[j] < e1[j] else 0
                if below >= 3:
                    return dict(exit_mi=int(mi[j + 1]), exit_px=o[j + 1], gross=o[j + 1] / px - 1, reason="9EMA 이탈")
    reason = "시간" if time_stop is not None and j_end < int(np.searchsorted(mi, until, side="right")) - 1 else "마감"
    return dict(exit_mi=int(mi[j_end]), exit_px=c[j_end], gross=c[j_end] / px - 1, reason=reason)


def _next_open(day, j):
    """j번 봉 신호 -> 다음 1분봉(3분 안) 번호, 없으면 None."""
    mi = day["mi"]
    if j + 1 >= len(mi) or mi[j + 1] > mi[j] + 3:
        return None
    return j + 1


def _trade(day, e, px, stop, max_stop, target_r=2.0, extra_target=None, **kw):
    if stop is None or px <= stop or (px - stop) / px > max_stop:
        return None
    tgt = px + target_r * (px - stop) if target_r else None
    if extra_target is not None:
        tgt = extra_target if tgt is None else min(tgt, extra_target)
    r = simulate(day, e, px, stop, tgt, **kw)
    if r is None:
        return None
    return dict(entry_mi=int(day["mi"][e]), entry_px=px, stop_dist=(px - stop) / px, **r)


# ---------------- b10: 급등주 그날 고가 돌파 ----------------
def b10(day, pc, adv_sh, frac, up=0.15, rv=5.0, t_hi=330, at_touch=False, target=True):
    mi, o, h, l, c, v = day["mi"], day["o"], day["h"], day["l"], day["c"], day["v"]
    j10 = int(np.searchsorted(mi, 30))  # 10:00 전 마지막 봉까지로 판단
    if j10 == 0 or j10 >= len(mi):
        return None
    p10 = c[j10 - 1]
    if p10 < pc * (1 + up) or not (1 <= p10 <= 50) or (c[:j10] * v[:j10]).sum() < 5e6:
        return None
    if cumvol_before(day, j10) < rv * adv_sh * frac(30):
        return None
    vw = vwap(day)
    k_h = int(np.argmax(h[:j10]))  # 지금까지 그날 고가 (j 전), 아래에서 이어서 갱신
    for j in range(j10, len(mi)):
        if j > j10 and h[j - 1] >= h[k_h]:
            k_h = j - 1
        if mi[j] > t_hi:
            break
        hod = h[k_h]
        if h[j] <= hod + TICK - 1e-9:
            continue
        # 고가를 넘는 봉: 쉬는 구간 조건
        rest = slice(k_h + 1, j)
        ok = mi[j] - mi[k_h] >= 15 and j > k_h + 1
        if ok:
            low = l[rest].min()
            ok = (hod - low) <= 0.4 * (hod - o[0]) and (c[rest] > vw[rest]).all()
        if not ok:
            continue
        stop = low - TICK
        if at_touch:
            px = max(o[j], hod + TICK)
            return _trade(day, j, px, stop, 0.06, 2.0 if target else None, ema5_exit=True, entry_in_bar=True)
        if c[j] > hod and v[j] >= 2 * v[:j + 1].mean():
            e = _next_open(day, j)
            if e is None or mi[e] > t_hi:
                return None
            return _trade(day, e, o[e], stop, 0.06, 2.0 if target else None, ema5_exit=True)
        # 넘었지만 거래량/종가 조건 미달 -> 고가가 바뀌었으니 계속 본다
    return None


# ---------------- b11: 둘째 날 이어가기 ----------------
def b11(day, line, target=True):
    mi, o, l, c, v = day["mi"], day["o"], day["l"], day["c"], day["v"]
    vw = vwap(day)
    for j in range(len(mi)):
        if mi[j] < 5:
            continue
        if mi[j] > 90:
            break
        if c[j] > line and v[j] >= 2 * v[:j + 1].mean():
            e = _next_open(day, j)
            if e is None:
                return None
            px = o[e]
            cands = [s for s in (l[:e].min() - TICK, vw[j] - TICK) if s < px]
            if not cands:
                return None
            return _trade(day, e, px, max(cands), 0.08, 2.0 if target else None, ema1_exit3=True)
    return None


# ---------------- b12: 첫 눌림 VWAP 반등 ----------------
def b12(day, pc, adv_sh, frac, up=0.10, line="vwap", cap2=True, target=True, t_hi=300):
    mi, o, h, l, c, v = day["mi"], day["o"], day["h"], day["l"], day["c"], day["v"]
    ref = vwap(day) if line == "vwap" else ema5_at_minute(day)[0]
    start = int(np.searchsorted(mi, 15))
    j0 = None
    for j in range(start, len(mi)):
        if np.isnan(ref[j]):
            return None
        if l[j] <= ref[j]:
            j0 = j
            break
    if j0 is None or mi[j0] > t_hi:
        return None
    for j in range(j0 + 1, len(mi)):
        if mi[j] - mi[j0] > 30 or mi[j] > t_hi:
            return None
        if c[j] > ref[j] and c[j] > h[j - 1]:
            px_now = c[j]
            if px_now < pc * (1 + up) or not (1 <= px_now <= 50) or (c[:j + 1] * v[:j + 1]).sum() < 5e6:
                return None
            if cumvol_before(day, j + 1) < 5 * adv_sh * frac(mi[j] + 1):
                return None
            hod = h[:j0].max()
            low = l[j0:j + 1].min()
            if hod - low > 0.6 * (hod - o[0]):
                return None
            e = _next_open(day, j)
            if e is None:
                return None
            px = o[e]
            stop = low - TICK
            if cap2:
                stop = max(stop, px * 0.98)
            tgt = None
            if target:
                tgt = px * 1.05 if hod <= px else min(px * 1.05, hod)
            return _trade(day, e, px, stop, 1.0, None, extra_target=tgt)
    return None


# ---------------- b13: 불 플래그 ----------------
def b13(day, pc, adv_sh, frac, pole_min=0.05, flag_len=(2, 6), lo_px=1.0, t_hi=120, time_stop=10):
    mi, o, h, l, c, v = day["mi"], day["o"], day["h"], day["l"], day["c"], day["v"]
    vw = vwap(day)
    n = len(mi)
    for p in range(1, n):
        if mi[p] > t_hi:
            break
        lo_k = max(0, p - 9)
        k_low = lo_k + int(np.argmin(l[lo_k:p + 1]))
        if k_low >= p:
            continue
        pole_lo, pole_hi = l[k_low], h[p]
        if pole_hi < pole_lo * (1 + pole_min) or h[p] < h[k_low:p + 1].max():
            continue
        pole_vol = v[k_low:p + 1].mean()
        for m in range(flag_len[0], flag_len[1] + 1):
            q = p + m  # 깃발 마지막 봉
            if q + 1 >= n or mi[q + 1] > t_hi or mi[q + 1] > mi[q] + 3:
                break
            fl = slice(p + 1, q + 1)
            f_hi, f_lo = h[fl].max(), l[fl].min()
            if f_hi >= pole_hi or f_lo < pole_hi - 0.5 * (pole_hi - pole_lo):
                break
            if v[fl].mean() >= 0.6 * pole_vol or f_lo <= vw[q]:
                continue
            j = q + 1
            if h[j] <= f_hi + TICK - 1e-9:
                continue
            px = max(o[j], f_hi + TICK)
            if not (lo_px <= px <= 20) or px < pc * 1.20 or (c[:j] * v[:j]).sum() < 3e6:
                break
            if cumvol_before(day, j) < 5 * adv_sh * frac(mi[j]):
                break
            stop = f_lo - TICK
            if px <= stop:
                break
            tgt = min(px + 2 * (px - stop), px + (pole_hi - pole_lo))
            return _trade(day, j, px, stop, 1.0, None, extra_target=tgt, time_stop=time_stop, entry_in_bar=True)
    return None


# ---------------- b14: 실적 둘째 날 ----------------
def b14(day, line, use_stop=True):
    mi, o, h, l = day["mi"], day["o"], day["h"], day["l"]
    vw = vwap(day)
    start = int(np.searchsorted(mi, 15))
    for j in range(start, len(mi)):
        if mi[j] > 90:
            break
        if h[j] > line + TICK - 1e-9:
            px = max(o[j], line + TICK)
            if not use_stop:
                r = simulate(day, j, px, None, entry_in_bar=True)
                return dict(entry_mi=int(mi[j]), entry_px=px, stop_dist=np.nan, **r)
            cands = [s for s in (l[:j].min() - TICK, vw[j - 1] - TICK) if s < px]
            if not cands:
                return None
            return _trade(day, j, px, max(cands), 0.06, None, entry_in_bar=True)
    return None


# ---------------- b15: 레드 투 그린 ----------------
def b15(day, pc, ph, adv_sh, frac, vol_mult=1.5, conservative=False, bar_stop=False, target=True):
    mi, o, h, l, c, v = day["mi"], day["o"], day["h"], day["l"], day["c"], day["v"]
    crossed, came_back = False, False
    for j in range(len(mi)):
        if mi[j] > 120:
            break
        if mi[j] < 5:
            continue
        sig = c[j] > pc and v[j] >= vol_mult * v[:j + 1].mean() and cumvol_before(day, j + 1) >= 3 * adv_sh * frac(mi[j] + 1)
        if conservative:
            if not crossed:
                crossed = c[j] > pc
                continue
            if not came_back:
                came_back = l[j] <= pc * 1.003
                if not came_back:
                    continue
                if not sig:
                    continue
            elif not sig:
                continue
        elif not sig:
            continue
        e = _next_open(day, j)
        if e is None:
            return None
        px = o[e]
        if px > pc * 1.02:
            return None
        stop = (l[j] if bar_stop else l[:e].min()) - TICK
        if px <= stop or (px - stop) / px > 0.04:
            return None
        tgt = max(ph, px + 2 * (px - stop)) if target else None
        r = simulate(day, e, px, stop, tgt)
        if r is None:
            return None
        return dict(entry_mi=int(mi[e]), entry_px=px, stop_dist=(px - stop) / px, **r)
    return None
