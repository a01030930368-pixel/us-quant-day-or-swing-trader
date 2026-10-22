# 워뇨띠(BitMEX aoa) 장부 분석·백테스트

장부 원본은 저장소에 넣지 않습니다 (600MB). 축약본은 프로젝트 파일 `crypto-wonyotti/data/`.

| 파일 | 하는 일 |
|---|---|
| `scripts/fetch_data.py` + `.github/workflows/crypto-data.yml` | 바이낸스 1분봉(BTC·ETH·XRP 2017~2026)을 받아 `crypto-data` 브랜치에 올림 |
| `analysis/episodes.py` | 체결내역 → 포지션 0→0 한 번의 매매로 묶기 (지갑 실현손익과 분기별 대조 완료) |
| `analysis/entry_context.py` | 진입 직전 등락·거래량·이평선 거리를 무작위 시점과 비교 |
| `analysis/replay.py` | 워뇨띠의 실제 진입에 기계적 청산을 붙여 보기 |
| `backtest/fade.py` | 역추세 지정가 규칙 시뮬레이터 (1분봉, 같은 봉 익절 금지 등 보수적 처리) |
| `backtest/grid.py` | 조건 조합 격자: 2018.03~2021.12 개발 / 2022.01~2026.09 검증, 수수료 4단계 |
