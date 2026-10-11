import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from swing import engine
from swing.insider import INSIDER, purchase_table, purchases_from_zip

from conftest import make_ohlc


def _zip() -> bytes:
    sub = ("ACCESSION_NUMBER\tFILING_DATE\tPERIOD_OF_REPORT\tDOCUMENT_TYPE\tISSUERTRADINGSYMBOL\n"
           "a1\t02-MAR-2020\t28-FEB-2020\t4\tbrk.b\n"
           "a2\t03-MAR-2020\t28-FEB-2020\t4/A\tAAA\n"
           "a3\t04-MAR-2020\t28-FEB-2020\t4\tAAA\n"
           "a4\t05-MAR-2020\t28-FEB-2020\t4\tAAA\n")
    trans = ("ACCESSION_NUMBER\tTRANS_CODE\tTRANS_SHARES\tTRANS_PRICEPERSHARE\tTRANS_ACQUIRED_DISP_CD\n"
             "a1\tP\t100\t50\tA\n"
             "a2\tP\t100\t50\tA\n"                 # amendment: skipped
             "a3\tS\t100\t50\tD\n"                 # a sale: skipped
             "a4\tP\t1000\t10\tA\n")
    owner = ("ACCESSION_NUMBER\tRPTOWNERCIK\tRPTOWNER_RELATIONSHIP\n"
             "a1\t1\tDirector\na2\t2\tOfficer\na3\t3\tOfficer\na4\t4\tTenPercentOwner\n")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("SUBMISSION.tsv", sub)
        z.writestr("NONDERIV_TRANS.tsv", trans)
        z.writestr("REPORTINGOWNER.tsv", owner)
    return buf.getvalue()


def test_only_insider_open_market_purchases():
    p = purchases_from_zip(_zip())
    assert list(p["ticker"]) == ["BRK-B"]           # fund-only owners, sales and amendments are out
    assert p["value"].iloc[0] == 5000
    idx = pd.bdate_range("2020-02-24", periods=20)
    tab = purchase_table(p, idx, ["BRK-B", "AAA"])
    assert tab.loc["2020-03-02", "BRK-B"] == 0 and tab.loc["2020-03-03", "BRK-B"] == 5000


@pytest.mark.parametrize("cls", INSIDER)
def test_insider_no_lookahead(cls):
    d1 = make_ohlc(n_days=1100, seed=6, drift=0.001)
    names = [c for c in d1["Close"].columns if c != "SPY"]
    rng = np.random.default_rng(2)
    buys = pd.DataFrame(0.0, index=d1["Close"].index, columns=names)
    for j in range(len(names)):
        days = rng.choice(np.arange(250, 1100), size=25, replace=False)
        buys.iloc[days, j] = rng.uniform(5e4, 5e5, size=25)
    d1["IBUY"] = buys
    d2 = {f: v.copy() for f, v in d1.items()}
    k = 800
    for f in ("Open", "High", "Low", "Close", "Volume"):
        d2[f].iloc[k + 1:] *= 1.5
    d2["IBUY"].iloc[k + 1:] *= 3
    r1 = engine.run(cls().prepare(d1), cost_bps=25)
    e2 = engine.run(cls().prepare(d2), cost_bps=25).equity
    pd.testing.assert_series_equal(r1.equity.iloc[:k + 1], e2.iloc[:k + 1])
    assert len(r1.trades) > 0
