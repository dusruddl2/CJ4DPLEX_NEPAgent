# demo/

대표 입력 하나로 NEPAgent의 실행 과정과 최종 결과를 단계별로 따라가는 자료입니다. 테스트 케이스 (b)(`results/qualitative/Q2_fb_0078_dense_resample/`)와 같은 영상이며, 아래 문장과 숫자는 모두 `nep_agent_output.json`에서 옮겼습니다.

| 파일 | 내용 |
|---|---|
| `video.mp4` | 입력 영상(관찰 구간 28초). 스쿠터를 배달받는 장면 |
| `question.json` | 질문과 보기. 보기는 예측 단계에 들어가지 않고 채점에만 씁니다 |
| `nep_agent_output.json` | 단계별 출력 요약 |
| `figure.png` | 실행 과정을 한 장으로 정리한 그림 |

## 실행 과정

![실행 과정](figure.png)

| 단계 | 출력 |
|---|---|
| 1. 초기 예측 | "The man pushes the blue scooter along the street, moving it from the truck bed towards the house, with the child watching him." (남자가 스쿠터를 집 쪽으로 밀고 감) |
| 2. 확신도 게이트 | 확신도 0.657 < 0.75 → 추가 단계로 진행 |
| 3. 도구 선택 | 구간 재샘플링, 구간 `#27`–`#32`, 확인할 대상 "말하고 있는 노란 옷의 여성과 파란 유니폼의 여성" |
| 4. 도구 실행 | 프레임 5장 추가 (`#27.1`, `#28.1`, `#29.1`, `#30.1`, `#31.1`) |
| 5. 인과 추론 | 마지막 상태 "남자가 스쿠터를 밀고 있고 아이가 지켜봄" → 인과 전이 "스쿠터를 옮긴 뒤 소개 장면으로 이어짐" → 예측 "스쿠터 클로즈업에 이어, 파란 유니폼의 여성이 전시장에서 카메라를 보고 설명함". 근거 프레임 `#27`, `#27.1` |
| 6. Matcher | 예측 문장 → 보기 C(정답) |

초기 예측은 지금 보이는 장면을 그대로 서술해 오답(A)이었고, 끝 구간을 촘촘히 다시 본 뒤의 예측이 정답(C)이 되었습니다. 이 기록은 구간 재샘플링만 쓰도록 한 실행에서 가져왔습니다.

## 직접 실행하기

모델 준비(`README.md` 4절)를 마친 뒤 아래 명령으로 같은 영상에 Agent를 실행할 수 있습니다. 결과는 `results/single/`에 JSON과 요약 그림으로 저장됩니다.

```bash
PYTHONPATH=. python scripts/run_agent.py --video demo/video.mp4
```
