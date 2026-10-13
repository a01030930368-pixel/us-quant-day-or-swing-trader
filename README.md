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
| `swing/refine.py` | 와인스타인·다바스 다듬기: 와인스타인 5종목 집중, 두 전략을 한 계좌에서 함께 운용 |
| `swing/fundamentals.py` | 미국 증권거래위원회(SEC) 무료 공시에서 분기 EPS·매출을 받아, 공시 다음 날부터만 쓰는 실적 성장 표시 |
| `swing/earnings_legends.py` | 미너비니·오닐·와인스타인·다바스에 오닐식 실적 조건(분기 EPS·매출 +25%)을 붙인 버전 |
| `swing/quality.py` | 총이익률(매출총이익 ÷ 총자산) 상위 회사를 몇 달씩 보유하는 장기 전략 3가지 |
| `swing/announce.py` | SEC 실적 발표 공시(8-K) 날짜 기준 실적 서프라이즈와 발표 반응 전략 3가지 |
| `swing/insider.py` | SEC 내부자 거래 데이터(Form 4)로 임원·이사의 장내 매수를 따라가는 전략 3가지 |
| `swing/ptj.py` | 유튜브 '폴 튜더 존스식 변곡점 돌파 + 5일선 추격매수' 규칙서 02의 시험 목록 J1~J5 (나눠 사기 포함) |
| `swing/chartdoc.py` | 유튜브 차트박사 '신고가 + 20일선 눌림 + 5일선 돌파' 규칙서 03의 시험 목록 S1~S7 |
| `swing/velez.py` | 유튜브 올리버 벨레즈 '3연속 음봉 눌림 + 전봉 고가 돌파' 규칙서 04의 시험 목록 V1~V6 |
| `swing/pullback_yt.py` | 유튜브 '눌림목 매매'(진짜 트레이딩) 규칙서 01의 시험 목록 P1~P5 |
| `swing/sewer.py` | 유튜브 '하수구 돌파' 매매법: 오래 200일선 아래 있던 종목이 첫 돌파 실패 후 더 큰 거래량으로 다시 돌파할 때 사는 전략 3가지 |
| `swing/blend.py` | 여러 전략을 한 계좌에 나눠 담고 매달 비중을 다시 맞추는 혼합 계산 |
| `swing/reversion.py` | 단기 과매도 반등: 상승 추세 종목이 며칠 급락하면 지정가로 사서 반등하면 파는 공개 전략 3가지 |
| `swing/pead.py` | 실적 서프라이즈 후 추세(PEAD): 분기 EPS가 평소보다 크게 늘면 공시 다음 날 사서 60거래일 보유 |
| `swing/engine.py` | 거래 단위 백테스트 (다음 날 시가 체결, 장중 손절, 거래비용, 미래 데이터 사용 방지) |
| `swing/universe.py` | 매매 대상 종목 목록: 대형주 약 100개, 중형주 약 130개, 소형주 약 160개 |
| `swing/metrics.py` | 수익률·최대낙폭·샤프 계산 (us-quant-trader에서 가져옴) |
| `swing/brokers.py` | 가짜 계좌(SimBroker)와 키움 REST 연결 (us-quant-trader에서 가져옴, 모의매매 단계에서 사용) |
| `scripts/run_backtest.py` | 전략을 백테스트하고 `reports/swing_report_*.md` 생성 (`--universe large` 대형주, `all` 대형주+중형주, `small` 소형주, `every` 전부, `sp1500` S&P 1500 전체(위키백과에서 목록 받음); `--set legends` 1그룹, `legends2` 2그룹, `legends_all` 1·2그룹, `refine` 와인스타인·다바스 다듬기, `earnings` 실적 조건 비교, `earnings_wide` 오닐·미너비니 + 실적(넓은 종목용), `pead` 실적 서프라이즈 후 추세, `reversion` 단기 과매도 반등, `quality` 총이익률 장기 보유, `announce` 실적 발표일 기준 서프라이즈, `insider` 내부자 매수 따라가기, `blend` 여러 전략 합치기, `sewer` 하수구 돌파(200일선 두 번째 돌파), `pullback_yt` 유튜브 눌림목 매매(규칙서 01), `ptj` 유튜브 폴 튜더 존스식 변곡점 돌파(규칙서 02), `chartdoc` 유튜브 차트박사 신고가 + 20일선 눌림 + 5일선 돌파(규칙서 03), `velez` 유튜브 벨레즈 3연속 음봉 눌림 + 전봉 고가 돌파(규칙서 04), `round2` 1·2차 후보) |

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
