# us-quant-day-or-swing-trader

미국주식 스윙 트레이딩(며칠~몇 주 보유) 자동매매 앱입니다.

진행 순서: 백테스트로 전략 고르기 → 앱 자체 모의매매 → 직접 켰을 때만 소액 실거래.
수익을 쫓기보다 전략 검증, 손실 통제, 안정적인 주문을 먼저 챙깁니다.

## 구성

| 경로 | 내용 |
|---|---|
| `swing/data.py` | 야후 파이낸스 일봉(시가·고가·저가·종가·거래량) 다운로드 (캐시) |
| `swing/indicators.py` | 이동평균, RSI, ATR |
| `swing/strategies.py` | 스윙 후보 전략: 주간 모멘텀, 실적 갭 상승 추종, 52주 신고가 근접 (1차에서 탈락한 눌림목·신고가 돌파도 참고용으로 보관) |
| `swing/legends.py` | 전설 4인 규칙서(와인스타인·미너비니·본드·오닐)를 숫자 그대로 옮긴 전략. 주석의 [원전]/[해석]/[근사]는 문서 12장 표시와 같음 |
| `swing/legends2.py` | 전설 2그룹(다바스·쿨라매기·래쉬키 홀리 그레일/터틀 수프·켈). 래쉬키는 원전대로 돌파 가격 예약 매수 |
| `swing/engine.py` | 거래 단위 백테스트 (다음 날 시가 체결, 장중 손절, 거래비용, 미래 데이터 사용 방지) |
| `swing/universe.py` | 매매 대상 종목 목록: 대형주 약 100개, 중형주 약 130개 |
| `swing/metrics.py` | 수익률·최대낙폭·샤프 계산 (us-quant-trader에서 가져옴) |
| `swing/brokers.py` | 가짜 계좌(SimBroker)와 키움 REST 연결 (us-quant-trader에서 가져옴, 모의매매 단계에서 사용) |
| `scripts/run_backtest.py` | 전설 4인 전략을 백테스트하고 `reports/swing_report_<universe>.md` 생성 (`--universe large` 대형주, `all` 대형주+중형주, `--set round2`는 1·2차 후보) |

## 실행 방법

```bash
pip install -r requirements.txt
python scripts/run_backtest.py     # 백테스트 + 보고서
pytest -q                          # 테스트
```

PC 설치 없이 돌리려면 GitHub의 **Actions → backtest → Run workflow**를 누르면 됩니다.

## 주의

- API 키는 `.env`에만 넣고 절대 커밋하지 마세요 (`.gitignore`에 포함됨).
- 종목 목록이 지금의 대형주라서 백테스트 결과가 실제보다 좋게 나올 수 있습니다(생존 편향).
