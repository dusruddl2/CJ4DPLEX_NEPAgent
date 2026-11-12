# 사용 내역 및 구현 범위

과제 문서 2.1(데이터)과 2.2(기존 기술·개발 도구)에서 요구한 사용 내역입니다. 라이선스는 2026-11-11에 각 공식 페이지에서 확인했고, 버전은 `requirements.txt`(추론), `requirements-train.txt`(학습) 기준입니다. 영상과 모델 가중치는 저장소에 넣지 않았습니다.

## 1. 데이터

| 데이터 | 출처 | 라이선스 | 용도 | 가공 |
|---|---|---|---|---|
| FutureOmni [2] | HF `OpenMOSS-Team/FutureOmni` | Apache-2.0 | 평가 (test 300문항 중 Speech 128) | 분할 시각까지 잘라 관찰 구간 생성, 보기 순서 섞기, Whisper 음성 전사 사전 계산 |
| FutureBench [1] | HF `haonan3/V1-33K` | Apache-2.0 | 평가 (test 200문항) | Charades·Ego4D 출처 제외, 보기 순서 섞기 |
| V1-33K [1] 학습 클립 | HF `haonan3/V1-33K` | Apache-2.0 | 학습 데이터 원천 | 아래 참조 |

- 라이선스는 주석(질문·보기·정답·캡션)에 대한 표기이며, 원본 영상의 권리는 각 원 출처에 있습니다.
- V1-33K의 기반 데이터 LLaVA-Video-178K [3]는 학술·교육 목적 사용만 허용합니다. 본 과제는 비상업 채용 과제이고 데이터를 재배포하지 않으므로 이 범위에 해당한다고 판단했습니다.
- **Charades·Ego4D는 이용 조건 때문에 학습·평가 모두에서 제외**했습니다(`EXCLUDED_SOURCES`, `tests/test_data_futurebench.py`). FutureBench 평가 영상도 학습 클립에서 제외했습니다.

**학습 데이터 생성 (가공·합성)**: V1-33K 클립에 Qwen3-VL-8B-Instruct를 Teacher로 적용해 추론 학습 데이터 1,500건을 만들었습니다(`src/train/build_sft_reasoning.py`).
1. Teacher 1단계: 정답 없이 프레임만 보고 관찰(마지막 상태, 진행 중 행동, 근거 프레임) 작성
2. Teacher 2단계: 정답 문장을 보고 관찰 이후의 중간 상태와 인과 전이만 작성
3. 규칙 필터로 형식 오류·정답 복사 등 제거. 1차 1,500개 중 997개 통과, 같은 풀에서 추가한 후보로 503개를 채움
4. 791건의 입력에 추론 시와 같은 코드로 도구 프레임 추가(`src/train/build_sft_tool_inputs.py`)

## 2. 모델

| 모델 | 라이선스 | 용도 | 직접 한 부분 |
|---|---|---|---|
| Qwen3-VL-8B-Instruct [7] | Apache-2.0 | 백본(전 단계), Teacher, 순수 Qwen 비교 대상 | LoRA 어댑터 학습·병합 |
| Whisper small [8] | MIT | 음성 전사 | 없음 |
| Grounding DINO tiny [10] | Apache-2.0 | 영역 확대 도구의 물체 탐지 | 상자 선택·여백 처리·실패 시 대체 |
| Qwen3-VL-32B-Instruct [7] | Apache-2.0 | Teacher 품질 비교에만 사용 (최종 시스템 미포함) | 없음 |
| PANNs CNN14 [9] | 가중치 CC-BY-4.0, 코드 MIT | 개발 단계 소리 태그 실험에만 사용 (최종 시스템 미포함) | 없음 |

직접 학습한 LoRA 어댑터(`checkpoints/nep_agent_lora`)는 저장소에 넣지 않았습니다.

## 3. 오픈소스

| 라이브러리 | 라이선스 | 용도 |
|---|---|---|
| vLLM 0.11.0 | Apache-2.0 | 로컬 추론 서빙 |
| LLaMA-Factory 0.9.5.dev0, PEFT 0.18.1 | Apache-2.0 | LoRA 학습·병합 |
| transformers 4.57.1, qwen-vl-utils 0.0.14 | Apache-2.0 | 모델 로딩, 입력 전처리 |
| decord 0.6.0, ffmpeg | Apache-2.0, LGPL/GPL | 프레임 디코딩, 오디오 추출 |
| torch 2.8.0 | BSD-3-Clause | 실행 런타임 |

범용 라이브러리는 requirements 파일 참조.

## 4. 참고 연구

| 연구 | 가져온 것 | 다르게 한 것 |
|---|---|---|
| NEP [1] (코드 MIT) | 과제 정의, 벤치마크, SFT 데이터 변환 방식, 학습 하이퍼파라미터 | 전체 미세조정 대신 LoRA, 2단계 Teacher로 추론 데이터 생성 |
| ViKey [4] (지원자 참여 연구) | 프레임 번호 표시 방식 | 직접 재구현, 도구 프레임에 하위 번호 부여 |
| VepAgent [6] | 관찰 구간을 다시 보는 도구, 인과 전이 추론 | 도구는 규칙 기반, 도구 선택은 학습하지 않음 |
| Video-CoE [5] | Teacher로 추론 과정을 만들어 학습 | 정답은 2단계에서만 제공 |

## 5. 외부 API와 AI 도구

- **외부 API**: 사용하지 않았습니다. 모든 모델은 로컬에서 실행합니다.
- **Claude Code (Anthropic)**: 코드·실험 스크립트·문서 초안 작성 보조와 Teacher 데이터 감사(판정자)에 사용했습니다. 문제 정의, 설계 결정, 실험 선정, 결과 검토는 지원자가 했습니다.

## 6. 직접 구현한 범위

위 모델·라이브러리를 제외한 다음 부분은 모두 직접 설계·구현했습니다.

| 영역 | 내용 | 경로 |
|---|---|---|
| Agent | 초기 예측, 확신도 게이트, 도구 선택·실행(3종), 인과 추론, 최종 답 선택, 구조화 출력 | `src/agent/`, `src/stages/`, `src/nep_agent/` |
| 학습 | 2단계 Teacher 프롬프트·필터, 도구 프레임 입력 생성, 학습 설정 | `src/train/` |
| 평가 | 데이터 로딩·분할, Matcher, 정확도 계산, 표 생성 | `src/data/`, `src/eval/`, `scripts/` |
| 테스트 | 미래 정보 누출 검사, 데이터·파이프라인 테스트 | `tests/` |

## 7. 실행 환경

- 학습: A100 80GB 1장, 약 6.9시간(1,500건 × 3 epoch)
- 추론: A100 80GB 또는 RTX A6000 48GB 1장
- Linux, Python 3.11, CUDA 12.8

## 참고 문헌

- [1] H. Wang et al., "Fostering Video Reasoning via Next-Event Prediction", arXiv:2505.22457.
- [2] Q. Chen et al., "FutureOmni: Evaluating Future Forecasting from Omni-Modal Context for Multimodal LLMs", arXiv:2601.13836.
- [3] Y. Zhang et al., "LLaVA-Video: Video Instruction Tuning With Synthetic Data", arXiv:2410.02713.
- [4] Y. Lee et al., "ViKey: Enhancing Temporal Understanding in Videos via Visual Prompting", arXiv:2603.23186.
- [5] Q. Su et al., "Video-CoE: Reinforcing Video Event Prediction via Chain of Events", arXiv:2603.14935.
- [6] Q. Chen et al., "VepAgent: Bridging Causal-Transition via Tool-Augmented Reinforcement Learning for Video Event Prediction", arXiv:2610.06293.
- [7] S. Bai et al., "Qwen3-VL Technical Report", arXiv:2511.21631.
- [8] A. Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision", arXiv:2212.04356.
- [9] Q. Kong et al., "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition", arXiv:1912.10211.
- [10] S. Liu et al., "Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection", arXiv:2303.05499.
