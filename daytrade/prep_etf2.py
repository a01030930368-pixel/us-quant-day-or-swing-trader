"""규칙서 b34·b35용 자료 뽑기: 섹터 SPDR 9개와 채권·금·해외 ETF의 1분봉 (prep_etf.py와 같은 방식, 종목만 다름).

prep_etf.py 캐시를 건드리지 않으려고 따로 둔다 -> prep_etf2/bars_YYYY.parquet
"""
import prep_etf

prep_etf.TICKERS = ("XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "TLT", "GLD", "EFA", "EEM")

if __name__ == "__main__":
    prep_etf.main()
