"""워뇨띠(aoa) BitMEX 체결내역에서 '포지션 0 → 0' 구간(한 번의 매매)을 복원합니다.

입력: ex.parquet (축약 체결내역), 출력: episodes.parquet
손익 계산
- 인버스(XBTUSD, XBT 선물): 손익(XBT) = Σ s·q/p   (s: 매수 +1, 매도 -1)
- 그 외(퀀토·선형): 손익(정산통화) ∝ -Σ s·q·p  (배율은 지갑 실현손익으로 맞춤)
"""
import sys
import numpy as np
import pandas as pd

src, out = sys.argv[1], sys.argv[2]
d = pd.read_parquet(src)
d = d[d.exectype == "Trade"].sort_values(["symbol", "t"], kind="stable")
d["s"] = np.where(d.side == "Buy", 1, -1)
inv = d.symbol.str.match(r"^XBT(USD|[HMUZ]\d\d)$")

rows = []
for sym, g in d.groupby("symbol", sort=False):
    s = g.s.to_numpy(); q = g.lastqty.to_numpy().astype(float); p = g.lastpx.to_numpy()
    t = g.t.to_numpy(); oid = g.orderid.to_numpy(); maker = (g.lastliquidityind == "M").to_numpy()
    ordt = g.ordtype.to_numpy(); comm = g.execcomm.to_numpy().astype(float)
    is_inv = bool(inv.loc[g.index[0]])
    pos = 0.0; start = None; acc = None
    for i in range(len(g)):
        dq = s[i] * q[i]
        newpos = pos + dq
        # 부호가 뒤집히면 체결을 둘로 나눔 (청산분 / 신규분)
        if pos != 0 and np.sign(newpos) != np.sign(pos) and newpos != 0:
            parts = [(-pos), newpos]
        else:
            parts = [dq]
        for part in parts:
            if pos == 0 and part != 0:
                start = i
                acc = dict(symbol=sym, side=int(np.sign(part)), t0=t[i], inv=is_inv,
                           xbt=0.0, lin=0.0, maxpos=0.0, n_fill=0, orders=set(),
                           buyq=0.0, buyv=0.0, sellq=0.0, sellv=0.0, makerq=0.0, comm=0.0,
                           stopq=0.0, adds=0)
            a = acc
            a["xbt"] += part / p[i]
            a["lin"] += -part * p[i]
            a["n_fill"] += 1; a["orders"].add(oid[i]); a["comm"] += comm[i] * abs(part) / q[i]
            if part > 0: a["buyq"] += part; a["buyv"] += part * p[i]
            else: a["sellq"] += -part; a["sellv"] += -part * p[i]
            if maker[i]: a["makerq"] += abs(part)
            if ordt[i] in ("Stop", "StopLimit"): a["stopq"] += abs(part)
            pos += part
            a["maxpos"] = max(a["maxpos"], abs(pos))
            if abs(pos) < 1e-9:
                pos = 0.0
                a["t1"] = t[i]; a["n_orders"] = len(a.pop("orders"))
                rows.append(a); acc = None
    if acc is not None:  # 끝까지 안 닫힌 포지션
        acc["t1"] = pd.NaT; acc["n_orders"] = len(acc.pop("orders")); acc["open"] = True
        rows.append(acc)

e = pd.DataFrame(rows)
e["entry_px"] = np.where(e.side > 0, e.buyv / e.buyq, e.sellv / e.sellq)
e["exit_px"] = np.where(e.side > 0, e.sellv / e.sellq.replace(0, np.nan), e.buyv / e.buyq.replace(0, np.nan))
e["ret"] = e.side * (e.exit_px / e.entry_px - 1)        # 평균단가 기준 수익률(레버리지 없이)
e["hold_min"] = (e.t1 - e.t0).dt.total_seconds() / 60
e["maker_share"] = e.makerq / (e.buyq + e.sellq)
e["stop_share"] = e.stopq / (e.buyq + e.sellq)
e.to_parquet(out)
print(len(e), "episodes")
