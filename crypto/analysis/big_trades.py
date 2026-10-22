"""큰 매매(수익 상위 40 + 손실 상위 10)를 차트에 표시한 PDF와 상황 표를 만듭니다.

각 칸: 1시간 종가(진입 7일 전 ~ 청산 2일 후), 그의 체결(매수 ▲ 초록, 매도 ▼ 빨강, 크기 = 수량),
보유 구간 음영, 아래 칸은 포지션 크기(백만 달러).
사용: python big_trades.py <데이터폴더> <작업폴더> <출력.pdf> <출력.csv>
"""
import glob, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages

font_manager.fontManager.addfont("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
plt.rcParams["axes.unicode_minus"] = False

data, work, out_pdf, out_csv = sys.argv[1:5]
px = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(f"{data}/BTCUSDT_1m_20*.parquet")))
px = px.set_index("t").sort_index()
c1h = px.c.resample("1h").last().ffill()
f5 = pd.read_parquet(f"{work}/feat5.parquet")
e = pd.read_parquet(f"{work}/episodes.parquet")
e = e[(e.symbol == "XBTUSD") & e.t1.notna()].reset_index(drop=True)
d = pd.read_parquet(f"{work}/ex.parquet", columns=["symbol", "exectype", "side", "lastqty", "lastpx", "t"])
d = d[(d.symbol == "XBTUSD") & (d.exectype == "Trade")].sort_values("t", kind="stable")
pick = pd.concat([e.nlargest(40, "xbt").assign(kind="수익"), e.nsmallest(10, "xbt").assign(kind="손실")])

rows = []
with PdfPages(out_pdf) as pdf:
    for k in range(0, len(pick), 4):
        fig = plt.figure(figsize=(11.7, 16.5))
        gs = fig.add_gridspec(12, 1, hspace=0.6)
        for j, (_, r) in enumerate(pick.iloc[k:k + 4].iterrows()):
            a = r.t0 - pd.Timedelta(days=7); b = r.t1 + pd.Timedelta(days=2)
            ax = fig.add_subplot(gs[3 * j:3 * j + 2, 0]); ax2 = fig.add_subplot(gs[3 * j + 2, 0], sharex=ax)
            s = c1h.loc[a:b]
            ax.plot(s.index, s.values, color="#555", lw=1)
            ax.axvspan(r.t0, r.t1, color="#cfe3ff", alpha=0.6)
            g = d[(d.t >= r.t0) & (d.t <= r.t1)].copy()
            g["q"] = g.lastqty * np.where(g.side == "Buy", 1, -1)
            gg = g.groupby([g.t.dt.floor("1h"), "side"]).agg(q=("lastqty", "sum"), p=("lastpx", "mean")).reset_index()
            mx = gg.q.max() if len(gg) else 1
            for sd, mk, col in [("Buy", "^", "#1a9850"), ("Sell", "v", "#d73027")]:
                x = gg[gg.side == sd]
                ax.scatter(x.t, x.p, s=10 + 200 * x.q / mx, marker=mk, color=col, alpha=0.7, zorder=3)
            pos = g.set_index("t").q.cumsum()
            ax2.step(pos.index, pos.values / 1e6, where="post", color="#2166ac")
            ax2.axhline(0, color="#999", lw=0.5)
            ax2.set_ylabel("포지션 $M", fontsize=8)
            side = "롱" if r.side > 0 else "숏"
            ax.set_title(f"[{r.kind}] {r.t0:%Y-%m-%d %H:%M} {side}  손익 {r.xbt:+.1f} BTC  "
                         f"평균단가 대비 {r.ret*100:+.1f}%  보유 {r.hold_min/1440:.1f}일  최대 ${r.maxpos/1e6:.0f}M  주문 {r.n_orders}개",
                         fontsize=9)
            ax.tick_params(labelsize=7); ax2.tick_params(labelsize=7)
            t5 = r.t0.floor("5min") - pd.Timedelta(minutes=5)
            ft = f5.loc[t5] if t5 in f5.index else pd.Series(dtype=float)
            sg = r.side
            rows.append(dict(kind=r.kind, t0=r.t0, side=side, pnl_btc=r.xbt, ret=r.ret, hold_d=r.hold_min / 1440,
                             maxpos_m=r.maxpos / 1e6, n_orders=r.n_orders,
                             r1h_s=ft.get("r1h", np.nan) * sg, r1d_s=ft.get("r1d", np.nan) * sg,
                             r7d_s=ft.get("r7d", np.nan) * sg, r30d_s=ft.get("r30d", np.nan) * sg,
                             pos7d=ft.get("pos7d", np.nan), pos30d=ft.get("pos30d", np.nan),
                             d_ma50d=ft.get("d_ma50d", np.nan), d_ma200d=ft.get("d_ma200d", np.nan),
                             vol1d=ft.get("vol1d", np.nan), v4h=ft.get("v4h", np.nan)))
        pdf.savefig(fig); plt.close(fig)
pd.DataFrame(rows).to_csv(out_csv, index=False)
