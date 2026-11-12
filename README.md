# NEPAgent: 영상 다음 사건 예측 Agent (Track 1)

입력 영상이 끝난 뒤에 일어날 사건을, 입력 영상만 보고 예측하는 Agent다. 평가 벤치마크는 FutureOmni [1]와 FutureBench [2]다.

처리 순서는 다음과 같다.

1. **관찰**: 입력 영상에서 프레임 32장을 균등 샘플링하고, 각 프레임 왼쪽 아래에 프레임 번호(`frame #01` ~ `frame #32`)를 그려 넣는다. 오디오는 Whisper [3] 음성 전사 텍스트로 바꿔 프롬프트에 넣는다. 효과음·음악 같은 비언어 소리는 다루지 않는다(`audio_tags: false`).
2. **Direct**(항상 실행): 다음 사건을 한 문장으로 예측하고, 그 문장의 확신도(생성 토큰 평균 로그확률의 지수)를 기록한다. 모델은 Qwen3-VL-8B [5]에 LoRA를 병합한 것이고 vLLM 엔진으로 프로세스 안에서 로드한다. 이 단계에서는 선택지를 보여주지 않는다.
3. **확신도 게이트**: 확신도가 임계값(τ = 0.75) 이상이면 Direct 답을 최종 답으로 쓰고 4, 5단계를 건너뛴다. 임계값 미만인 문항만 4, 5단계를 거친다. (`src/stages/routing.py`, `src/nep_agent/core.py`)
4. **도구 선택 및 실행**: 호출 한 번으로 도구 하나와 프레임 구간(최대 8장)을 고르고 실행한다. 최종 구성의 메뉴는 `terminal_state`(마지막 상태 재확인: 끝 3초에서 프레임 2장 추가), `dense_resample`(구간 재샘플링: 구간 안 프레임 추가), `zoom`(영역 확대: 구간 양 끝 프레임의 일부 확대)의 3종이다. 코드에는 `audio_segment`(음성 구간 읽기)도 있지만, 이미 입력에 있는 오디오 텍스트를 다시 가져오는 경우가 대부분이어서 최종 메뉴에서 뺐다. 도구 선택 호출은 오디오 텍스트를 본다. 도구 실행 자체는 VLM을 호출하지 않는다. (`src/stages/tool_select.py`, `src/agent/tools.py`)
5. **인과 추론**: 원래 프레임에 도구가 가져온 새 증거를 더해, 호출 한 번으로 마지막 프레임의 상태(terminal state), 진행 중인 행동, 근거 프레임, 영상 이후의 중간 상태, 인과 전이, 예측 사건을 JSON으로 출력한다(`tool_reason`). 인과 추론 모델은 오디오 텍스트 없이 학습했으므로, 두 추론 호출의 오디오 칸에는 학습 때와 같은 "Audio: not available."을 넣는다. 도구가 새 증거를 못 가져오거나 이 JSON이 깨지면 원래 프레임만으로 같은 추론을 한 번 실행한다(`causal_sft_nc`). 그 답도 없으면 Direct 답을 쓴다. (`src/nep_agent/`)
6. **Matcher**: 예측 문장을 선택지에 대응시킨다. 이미지 없이 텍스트만 쓰며, 선택지를 읽는 단계는 여기뿐이다. 선택지를 주지 않으면 이 단계를 건너뛰고 예측 문장과 추론 기록만 낸다. (`src/stages/matcher.py`)

임계값은 `configs/nep_agent.yaml`의 `gate_tau: 0.75`다. 확신도가 임계값 이상인 문항은 Direct와 Matcher만 실행하고 끝난다. 최종 답이 어디에서 왔는지는 결과 JSON의 `trace.nep_agent.final_source`에 남는다.

모델은 도구가 추가한 프레임이 들어간 입력으로 학습한 LoRA다(8절). 설계 근거와 실험 결과는 `docs/Technical_Report.md`에 있다.

## 1. 실행 환경 및 요구사항

| 항목 | 내용 |
|---|---|
| OS | Linux x86_64 (`requirements.txt` 주석 기준) |
| Python | 3.11 |
| CUDA | 12.8 (`requirements.txt` 주석 기준, `torch==2.8.0`) |
| GPU | 추론은 GPU 1장. vLLM이 그 GPU 메모리의 `VLM_GPU_FRACTION`(기본 0.6)만큼을 잡고, 단일 영상 실행에서는 Whisper와 PANNs도 같은 GPU에 올라간다. |
| GPU 메모리 | 병합한 8B 모델을 48GB와 80GB GPU에서 `VLM_GPU_FRACTION` 0.38~0.8로 실행했다. 우리 실행에서 가장 작았던 할당은 약 30GB이고, 대부분의 실행은 48GB GPU에서 했다. 이 값은 실행해 본 가장 작은 할당이며 최소 요구량을 잰 것은 아니다. |
| ffmpeg | 시스템에 `ffmpeg`가 있으면 그것을 쓰고, 없으면 `imageio-ffmpeg`에 포함된 바이너리를 쓴다. 따로 설치하지 않아도 된다. |
| 디스크 | FutureOmni 테스트 영상 압축 파일이 약 46 GiB다(`scripts/download_data.sh`에 적힌 파트 크기 합). 두 벤치마크 2,000문항의 캐시(번호를 그린 프레임, 오디오 텍스트, 잘라 낸 영상)는 우리 실행에서 약 54GB였다. |

학습 재현(8절)과 어댑터 병합(3.4)은 별도 가상환경 `.venv_train`을 쓴다. 의존성은 `requirements-train.txt`에 있다(LLaMA-Factory, `peft`).

## 2. 환경 구성 및 의존성 설치 방법

저장소 루트에서 실행한다. `scripts/download_data.sh`가 기본으로 `.venv/bin/hf`를 찾으므로 가상환경 이름은 `.venv`로 둔다.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

`scripts/` 아래 실행 스크립트는 `src` 패키지를 import하므로 저장소 루트에서 `PYTHONPATH=.`을 붙여 실행한다.

환경 변수는 셸에서 `export`하거나 저장소 루트의 `.env` 파일에 적는다(`src/models/vlm.py`가 `python-dotenv`로 읽는다). 전체 목록은 10절에 있다.

## 3. 외부 모델·체크포인트 준비

### 3.1 Qwen3-VL-8B-Instruct (기반 VLM)

| 항목 | 내용 |
|---|---|
| 명칭 | Qwen3-VL-8B-Instruct |
| 다운로드 위치 | Hugging Face `Qwen/Qwen3-VL-8B-Instruct` |
| 준비 방법 | 로컬 디렉터리로 내려받는다(아래 명령). |
| 경로 설정 | 환경 변수 `VLM_PATH`에 그 디렉터리를 지정한다. 지정하지 않으면 `Qwen/Qwen3-VL-8B-Instruct`를 Hub 이름 그대로 쓴다. 실행 스크립트의 `--model` 인자로도 바꿀 수 있다. |

```bash
hf download Qwen/Qwen3-VL-8B-Instruct --local-dir models/Qwen3-VL-8B-Instruct
export VLM_PATH=$PWD/models/Qwen3-VL-8B-Instruct
```

LoRA 병합(3.4)은 기반 모델의 `config.json`과 토크나이저 파일을 디렉터리에서 직접 읽으므로, 병합을 하려면 `VLM_PATH`가 로컬 디렉터리여야 한다.

### 3.2 Whisper (음성 전사)

| 항목 | 내용 |
|---|---|
| 명칭 | OpenAI Whisper `small` (`openai-whisper` 패키지) |
| 다운로드 위치 | `openai-whisper`가 정한 공식 배포 주소 |
| 준비 방법 | 따로 할 일이 없다. `whisper.load_model("small", device="cuda")`가 처음 호출될 때 패키지가 자동으로 내려받는다(패키지 기본 캐시는 `~/.cache/whisper`). |
| 경로 설정 | 코드에 경로 설정이 없다. 모델 이름 `small`은 `src/models/audio.py`의 `AudioAnalyzer` 기본값이다. |

### 3.2b Grounding DINO tiny (영역 확대 도구)

| 항목 | 내용 |
|---|---|
| 명칭 | `IDEA-Research/grounding-dino-tiny` (Hugging Face, Apache-2.0) |
| 준비 방법 | 따로 할 일이 없다. 영역 확대 도구가 처음 실행될 때 transformers가 Hugging Face Hub에서 내려받는다. 미리 받으려면 `hf download IDEA-Research/grounding-dino-tiny` |
| 경로 설정 | 환경 변수 `ZOOM_DETECTOR`로 다른 모델 이름이나 로컬 경로를 줄 수 있다. 기본값은 위 모델 이름 |

### 3.3 PANNs CNN14 (최종 구성에서는 쓰지 않음)

오디오 캐시 생성 스크립트는 개발 단계와 같은 캐시 형식을 유지하려고 PANNs [4] 소리 태그도 함께 계산한다. 최종 구성(`configs/nep_agent.yaml`, `audio_tags: false`)은 이 태그를 프롬프트에 넣지 않는다. 태그를 넣은 ablation을 재현할 때만 의미가 있다.

| 항목 | 내용 |
|---|---|
| 명칭 | PANNs CNN14 (`panns-inference` 패키지의 `AudioTagging`) |
| 다운로드 위치 | `panns-inference`가 정한 배포 주소(Zenodo) |
| 준비 방법 | 따로 할 일이 없다. `AudioTagging(checkpoint_path=None, device="cuda")`로 부르므로 패키지가 기본 체크포인트와 라벨 파일을 `~/panns_data/`에 자동으로 내려받는다. 이 패키지는 다운로드에 `wget`을 쓴다. |
| 경로 설정 | 코드에 경로 설정이 없다(`checkpoint_path=None` 고정). |

오프라인 환경이면 3.2, 3.3의 파일을 위 기본 위치에 미리 넣어 둔다.

### 3.4 NEPAgent LoRA 어댑터 (지원자 학습)

| 항목 | 내용 |
|---|---|
| 명칭 | NEPAgent LoRA 어댑터 (`nep_agent_lora`). Qwen3-VL-8B-Instruct의 언어 모델 부분에 붙는 rank 64 LoRA다. |
| 다운로드 위치 | Google Drive: https://drive.google.com/drive/folders/1GoackN2mI0RrBRY4ba3xzBhlcIUMVR1i?usp=sharing (`nep_agent_lora.zip`: 어댑터, `nep_agent_lora_merged.zip`: 병합 모델). 제출 저장소에는 포함되어 있지 않다(`checkpoints/`는 `.gitignore`로 제외한다). 압축 해제 방법은 `checkpoints/README.md` 참조 |
| 준비 방법 | 내려받은 어댑터 폴더(`adapter_model.safetensors`, `adapter_config.json`, 토크나이저·전처리 설정 파일, 학습 기록 `train_results.json`, `trainer_log.jsonl` 포함)를 `checkpoints/nep_agent_lora/`에 두고, 아래 명령으로 기반 모델에 병합한다. |
| 경로 설정 | `configs/nep_agent.yaml`, `configs/nep_agent_all_candidates.yaml`의 `lora_path: checkpoints/nep_agent_lora`가 이 위치를 가리킨다. 실제로 로드하는 모델은 `<lora_path>_merged`, 즉 `checkpoints/nep_agent_lora_merged/`다(`src/models/vlm.py`의 `resolve_model_path`). 이 폴더가 없으면 실행이 `FileNotFoundError`로 멈춘다. |

병합 명령(`.venv_train` 환경, 구성 방법은 8절):

```bash
PYTHONPATH=. .venv_train/bin/python -m src.train.merge_adapter \
  --base "$VLM_PATH" \
  --adapter checkpoints/nep_agent_lora \
  --out checkpoints/nep_agent_lora_merged
```

- 추가 인자는 `--device`(기본 `cpu`)와 `--force`(기존 출력 폴더 덮어쓰기)다.
- 병합에는 `peft`가 필요하다. `peft`는 `requirements.txt`에는 없고 `requirements-train.txt`에 있다(`peft==0.18.1`, 어댑터의 `adapter_config.json`에 기록된 버전과 같다).
- `transformers` 버전이 4.57.1이 아니면 병합을 거부한다. 병합 뒤 `config.json`이 기반 모델과 달라졌는지도 검사한다.
- 병합된 모델 폴더(`checkpoints/nep_agent_lora_merged/`)도 제출 저장소에 들어 있지 않다. 위 명령으로 만든다. 병합된 모델 폴더를 직접 받은 경우에는 그것을 `checkpoints/nep_agent_lora_merged/`에 두면 되고 병합 단계는 건너뛴다.

## 4. Agent 실행 방법 및 실행 명령

GPU 지정은 `CUDA_VISIBLE_DEVICES`로 한다.

### 4.1 단일 영상

```bash
# 영상만 주는 경우: 파일 전체가 관찰 구간이고, 예측 문장과 추론 기록을 낸다
PYTHONPATH=. python scripts/run_agent.py \
  --video path/to/clip.mp4

# 앞 12초만 관찰 구간으로 주는 경우
PYTHONPATH=. python scripts/run_agent.py \
  --video path/to/clip.mp4 \
  --t_split 12

# 질문과 선택지를 함께 주는 경우: 예측 문장을 선택지에 대응시킨다
PYTHONPATH=. python scripts/run_agent.py \
  --video path/to/clip.mp4 \
  --t_split 12 \
  --question "What happens next?" \
  --options "He sits down" "He opens the door" "He leaves" "He falls"
```

| 인자 | 필수 | 기본값 | 설명 |
|---|---|---|---|
| `--video` | 예 | | 원본 영상 파일 |
| `--t_split` | | 파일 전체 | Agent가 보는 구간의 끝(초). 주지 않거나 영상 길이 이상의 값을 주면 파일 전체가 관찰 구간이다. 0 이하의 값은 오류다. |
| `--question` | | `What happens next, right after the clip ends?` | 질문 문장 |
| `--options` | | 없음 | 선택지. 주지 않거나 2~8개를 준다. 순서대로 A, B, C, ... 가 붙는다. |
| `--sample-id` | | 영상 파일 이름(확장자 제외) | 결과 파일 이름과 캐시 폴더 이름 |
| `--config` | | `configs/nep_agent.yaml` | 실행 설정 |
| `--model` | | `VLM_PATH` 값 | 기반 모델 |
| `--work-dir` | | `data/splits` | 잘라낸 영상을 두는 곳 |
| `--cache-root` | | `cache` | 프레임·오디오 캐시 |
| `--output-dir` | | `results/single` | 결과 저장 위치 |

`--t_split`이 영상 길이보다 작으면 스크립트는 0초부터 그 시점까지만 잘라 `<work-dir>/<sample-id>/prefix.mp4`를 만들고 Agent에는 이 파일만 넘긴다. 분할 시점 이후 구간은 파일로 만들지 않는다. `--t_split`을 주지 않거나 영상 길이 이상이면 아무것도 자르지 않고 원본 파일을 그대로 관찰 구간으로 쓴다. Whisper와 PANNs는 같은 프로세스에서 로드된다.

실행이 끝나면 결과 JSON과 요약 그림(`.png`)을 저장하고, 판단 한 줄과 요약을 로그로 출력한다. 판단 한 줄은 Direct 확신도와 임계값, 최종 답의 출처를 말한다. 예: `Direct confidence 0.65 is below the threshold 0.75; reported answer: tool_reason`, `Direct answer kept: its confidence 0.89 is at or above the threshold 0.75; no tool or reasoning call was made`. 요약에는 `prediction`, `final_source`, 최종 답의 추론 기록, 사용한 도구와 그 결과가 들어 있다.

요약 그림(`src/agent/figure.py`)에는 위에서부터 균등 샘플링한 프레임 가운데 12장의 띠(근거로 인용된 프레임은 빨간 테두리), 도구가 추가한 프레임(파란 테두리)과 띠에 없는 인용 프레임, 그리고 글이 나온다. 글은 Direct 예측 문장, Direct 확신도와 임계값 및 그에 따른 분기, 도구와 구간과 실행 결과, 오디오 도구가 가져온 텍스트, 마지막 상태, 진행 중인 행동, 인과 전이, 근거 프레임, 예측 사건, 최종 답의 출처, 선택지별 Matcher 확률이다. 도구 뒤 추론의 JSON 실패 여부는 그림에 나오지 않고 결과 JSON에 있다.

선택지 유무에 따른 차이:

- 선택지 없음: Matcher를 호출하지 않는다. `prediction.next_event`에 예측 문장이 들어가고 `prediction.answer`, `prediction.confidence`, `prediction.uncertain`은 `null`, `prediction.option_probs`와 `prediction.top_k`는 비어 있다. 추론 기록은 `trace.nep_agent.reasoning`에 저장된다. 기본 설정에서 확신도가 임계값 이상인 영상은 Direct 문장만 내고 추론 기록은 없다. 요약 그림은 선택지 부분만 빼고 같은 형식으로 저장된다.
- 선택지 있음: 각 후보의 예측 문장을 Matcher가 선택지에 대응시킨다. 요약 그림에 선택지별 확률이 함께 나온다.

설정별 차이:

- `configs/nep_agent.yaml`(기본): Direct를 먼저 실행한다. 확신도가 임계값(τ = 0.75) 이상이면 Direct 답이 `prediction`이고 다른 단계는 실행하지 않는다. 임계값 미만이면 도구 선택·실행과 인과 추론(`tool_reason`)을 실행하고, 그 답이 없으면 원래 프레임만 본 인과 추론(`causal_sft_nc`)을, 그것도 없으면 Direct 답을 `prediction`으로 낸다. 실행한 단계의 기록만 `trace.nep_agent.reasoning`에 남는다.
  추론 호출의 오디오 칸은 "Audio: not available."(`reason_audio: none`), 도구 메뉴는 3종(`tool_menu: dense_resample,zoom,terminal_state`)이다.
- `configs/nep_agent_all_candidates.yaml`: 정책 비교용. 같은 설정으로 모든 문항에 Direct, 인과 추론(`causal_sft_nc`), 도구 사용 후 인과 추론(`tool_reason`) 세 후보를 모두 계산해 저장하고(`store_all_candidates: true`) `prediction`은 Direct 답이다. 저장된 후보에 임계값(τ = 0.75)을 적용하면 `configs/nep_agent.yaml`과 같은 답이 된다.

`run_agent.py`에는 설정 값을 덮어쓰는 인자가 없다. 값을 바꾸려면 YAML을 고치거나 4.2의 `--set`을 쓴다.

### 4.2 벤치마크

데이터 준비(5절)와 오디오 캐시 생성이 끝난 뒤에 실행한다. `run_benchmark.py`는 오디오 모델을 로드하지 않으므로, `use_audio: true`인 설정은 오디오 캐시가 없으면 오류로 멈춘다.

```bash
# 오디오 캐시 (Whisper 전사; PANNs 태그도 함께 저장되지만 최종 구성은 쓰지 않음, GPU)
PYTHONPATH=. python scripts/precompute_audio.py --split test
PYTHONPATH=. python scripts/precompute_audio.py --split fb_test

# NEPAgent (Direct 우선, 확신도가 임계값 미만인 문항만 도구 + 인과 추론)
PYTHONPATH=. python scripts/run_benchmark.py --config configs/nep_agent.yaml --split test
PYTHONPATH=. python scripts/run_benchmark.py --config configs/nep_agent.yaml --split fb_test

# 정책 비교용 (같은 설정으로 모든 문항에 세 후보를 모두 계산해 저장)
PYTHONPATH=. python scripts/run_benchmark.py --config configs/nep_agent_all_candidates.yaml --split test
PYTHONPATH=. python scripts/run_benchmark.py --config configs/nep_agent_all_candidates.yaml --split fb_test

# 도구 메뉴 ablation: 도구를 하나만 메뉴에 둠. 확신도가 임계값 미만인 문항만 담은 질문 파일에 실행한다.
# 질문 파일은 nep_agent_all_candidates 실행 결과에서 만든다 (정답 파일을 읽는 평가 쪽 스크립트).
PYTHONPATH=. python scripts/build_routed_splits.py --split test --audio-types Speech   # data/splits/questions_test_rt.jsonl
PYTHONPATH=. python scripts/build_routed_splits.py --split fb_test                     # data/splits/questions_fb_test_rt.jsonl
PYTHONPATH=. python scripts/run_benchmark.py --config configs/ablation/tools_only_terminal.yaml --split test_rt
PYTHONPATH=. python scripts/run_benchmark.py --config configs/ablation/tools_only_dense.yaml --split test_rt
PYTHONPATH=. python scripts/run_benchmark.py --config configs/ablation/tools_only_zoom.yaml --split test_rt

# 베이스라인 (학습 없는 Qwen3-VL-8B)
PYTHONPATH=. python scripts/run_benchmark.py --config configs/plain_qwen.yaml --split test
PYTHONPATH=. python scripts/run_benchmark.py --config configs/plain_qwen.yaml --split fb_test
```

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--config` (필수) | | 설정 파일 하나 이상. 한 프로세스는 모델 하나만 로드하므로 `lora_path`가 같은 설정끼리만 함께 줄 수 있다. |
| `--split` (필수) | | `<splits-root>/questions_<split>.jsonl`을 읽는다. 5절의 스크립트가 만드는 이름은 `dev`, `test`, `test_original`, `full`(FutureOmni), `fb_dev`, `fb_test`, `fb_full`(FutureBench)이다. |
| `--set` | | 설정 덮어쓰기 `key=value` (값은 JSON으로 해석하고, 실패하면 문자열). 예: `--set run_name=nep_agent_rerun`은 결과를 다른 폴더에 저장한다. |
| `--shard` | `0/1` | `i/n` 형식. 문항을 n개로 나눠 i번째만 실행한다(GPU 여러 장에 프로세스를 나눠 띄울 때). |
| `--model` | `VLM_PATH` 값 | 기반 모델 |
| `--splits-root` | `data/splits` | 질문 파일과 `prefix.mp4` 위치 |
| `--cache-root` | `cache` | 프레임·오디오 캐시 |
| `--results-root` | `results` | 결과 저장 위치 |
| `--batch-size` | 24 | 한 번에 처리하는 문항 수 |
| `--limit` | 0 | 앞에서 N개만 실행(0이면 전체) |

결과 파일이 이미 있는 문항은 건너뛰므로 중단된 실행은 같은 명령으로 이어서 돌릴 수 있다. `precompute_audio.py`의 인자는 `--split`(필수), `--shard`, `--splits-root`, `--cache-root`다.

설정 파일:

| 파일 | 모델 | 내용 |
|---|---|---|
| `configs/nep_agent.yaml` | `checkpoints/nep_agent_lora_merged` | NEPAgent(제출 모델, 최종). Direct 우선. 확신도가 임계값(`gate_tau: 0.75`) 미만인 문항만 도구 뒤 추론 → 원래 프레임만 본 추론 → Direct 순서로 답을 정한다. 추론 호출에 오디오 텍스트 없음(`reason_audio: none`), 도구 3종(`tool_menu`) |
| `configs/nep_agent_all_candidates.yaml` | `checkpoints/nep_agent_lora_merged` | 정책 비교용. 최종 구성과 같은 설정으로 모든 문항에 후보 세 가지(Direct, 원래 프레임만 본 추론, 도구 뒤 추론)를 저장하고 `prediction`은 Direct. 임계값(τ = 0.75)을 적용하면 `nep_agent.yaml`의 결과와 같다 |
| `configs/ablation/tools_only_*.yaml` | `checkpoints/nep_agent_lora_merged` | 도구 메뉴 ablation(`tools_only_terminal`, `tools_only_dense`, `tools_only_zoom`). 위 설정에서 메뉴에 도구 하나만 둠. 임계값 미만 문항만 담은 질문 파일(`_rt`)에 실행한다 |
| `configs/plain_qwen.yaml` | 기반 Qwen3-VL-8B | 베이스라인: 프레임 번호 없음, 오디오 없음, 같은 Direct 프롬프트와 Matcher |
| `configs/base.yaml` | | 공통 설정(프레임 32장, 프레임 번호, 오디오 텍스트). 다른 파일이 `inherit`로 상속한다. |

설정 키는 `src/schemas.py`의 `RunConfig`에 정의되어 있다. 최종 구성과 관련된 키는 `gate_tau`(임계값), `store_all_candidates`(모든 후보 저장 여부), `tool_menu`(도구 선택 호출에 보여 줄 도구, 비우면 전부), `reason_audio`(`none`이면 두 추론 호출의 오디오 칸을 "Audio: not available."로), `tool_note`(도구 뒤 추론 프롬프트에 도구가 추가한 프레임 번호를 적음, 최종 구성은 쓰지 않음)다.
| `configs/report_tables.yaml` | | 결과 표 스크립트의 행 정의 |

## 5. 입력 데이터 지정 방법

### 5.1 단일 영상

- **영상 파일**: `--video`에 준 파일. 코드에 확장자 제한은 없고 ffmpeg가 디코딩할 수 있는 형식이면 된다. 벤치마크 영상은 모두 mp4다.
- **자르는 지점**(생략 가능): `--t_split`(초). 영상 길이보다 작은 값을 주면 0초부터 이 시점까지를 ffmpeg로 `prefix.mp4`(H.264/AAC)로 다시 인코딩하고 Agent는 이 파일만 본다. 주지 않거나 영상 길이 이상이면 파일 전체를 그대로 본다. 벤치마크가 배포한 영상은 이미 분할 시점에서 끝나므로 `--t_split` 없이 넘긴다.
- **질문과 선택지**(둘 다 생략 가능): `--question`, `--options`. 질문을 생략하면 기본 질문을 쓴다. 선택지는 Matcher만 읽는다. Direct, 도구 선택, 인과 추론 프롬프트에는 선택지가 들어가지 않고, 질문에서 선택지를 가리키는 표현은 규칙으로 바꿔 쓴다(`src/stages/questions.py`).
- 오디오 트랙이 없는 영상은 오디오 없이 처리된다.

### 5.2 FutureOmni

```bash
# 1) 테스트 영상과 주석 다운로드 (Hugging Face OpenMOSS-Team/FutureOmni)
bash scripts/download_data.sh test

# 2) dev/test 질문 파일과 정답 파일 생성
PYTHONPATH=. python scripts/build_splits.py

# 3) 압축 파일에서 필요한 영상을 꺼내 data/splits/<sample_id>/prefix.mp4 생성
python scripts/prepare_prefix.py
```

- `download_data.sh`의 인자는 `test`, `train`, `all`(기본) 중 하나다. 이 저장소의 평가와 학습에는 `test`만 쓴다.
- 파일은 `$FUTUREOMNI_ROOT/test`에 저장되고(기본 `data/downloads/FutureOmni`), `data/raw/futureomni`가 그 폴더를 가리키는 심볼릭 링크로 만들어진다. `hf` 실행 파일이 `.venv/bin/hf`가 아니면 `HF_CLI`로 경로를 준다.
- `build_splits.py`는 `data/splits/questions_{dev,test,test_original}.jsonl`(Agent가 읽는 파일, 정답 없음)과 `data/labels/labels_*.jsonl`(평가용 정답)을 따로 쓴다. 기본 크기는 개발 세트 60, test 300이고 `--dev`, `--test`로 바꾼다. 선택지 순서는 고정 seed로 섞고, `test_original`은 원래 순서를 유지한다.
- `prepare_prefix.py`는 영상 길이가 주석의 자르는 시점보다 길면 그 시점에서 자른다. 처리 결과는 `logs/prepare_prefix_report.json`에 남는다.

### 5.3 FutureBench

FutureBench는 V1-33K 배포본에 들어 있다. 이 저장소에는 다운로드 스크립트가 없다. <!-- TODO: V1-33K 다운로드 URL과 받는 방법 --> 내려받은 폴더를 `V1_33K_ROOT`로 지정한다(기본 `data/raw/v1_33k`). 코드가 읽는 파일은 다음과 같다.

```
$V1_33K_ROOT/
├── futurebench.json                                  # FutureBench 문항
├── video_reasoning_future_prediction_v1.json         # 학습용 주석 (8절에서만 사용)
└── first_segment_video/first_part_video/<data_source>/<id>.mp4
```

```bash
export V1_33K_ROOT=/path/to/V1-33K

# 1) dev/test 질문 파일과 정답 파일 생성 (기본 dev 40, test 200)
python scripts/build_splits_fb.py

# 2) 영상은 이미 앞부분만 담고 있으므로 prefix.mp4로 링크
python scripts/prepare_prefix.py --benchmark futurebench
```

Charades와 Ego4D 출처 문항은 로더에서 제외한다(`src/data/futurebench.py`).

### 5.4 전체 문항 split

```bash
python scripts/build_full_splits.py
python scripts/prepare_prefix.py --questions data/splits/questions_full.jsonl \
  --report logs/prepare_prefix_full_report.json
python scripts/prepare_prefix.py --benchmark futurebench --questions data/splits/questions_fb_full.jsonl \
  --report logs/prepare_prefix_fb_full_report.json
```

`build_full_splits.py`는 개발 세트와 영상이 겹치는 문항을 뺀 전체 문항으로 `questions_full.jsonl`(FutureOmni), `questions_fb_full.jsonl`(FutureBench)을 만든다. 인자는 없다.

### 5.5 질문 파일 형식

`data/splits/questions_<split>.jsonl`의 한 줄:

```json
{"sample_id": "fo_0001", "video": "...", "t_split": 12.0, "question": "...", "options": [["A", "..."], ["B", "..."]]}
```

Agent는 이 파일과 `data/splits/<sample_id>/prefix.mp4`만 읽는다. 같은 형식으로 파일을 만들고 `prefix.mp4`를 두면 다른 데이터에도 `run_benchmark.py`를 쓸 수 있다.

## 6. 출력 결과의 저장 위치 및 확인 방법

| 실행 | 저장 위치 |
|---|---|
| 단일 영상 | `results/single/<sample_id>.json`과 같은 이름의 `.png`(요약 그림). 그림은 선택지가 없어도 저장된다. |
| 벤치마크 | `results/quantitative/runs/<split>/<run_name>/<sample_id>.json`. `run_name`은 설정 파일 이름(확장자 제외)이다. |
| 프레임·오디오 캐시 | `cache/<sample_id>/` (`raw/` 원본 프레임, `vp/` 번호를 그린 프레임, `zoom/` 확대 이미지, `audio_*.json`) |

확인 방법:

```bash
python -m json.tool results/single/<sample_id>.json | less
python -c "import json,sys; r=json.load(open(sys.argv[1])); print(r['status'], r['prediction']['answer'], r['prediction']['next_event'])" results/single/<sample_id>.json
```

주요 필드:

| 필드 | 의미 |
|---|---|
| `sample_id` | 문항 ID |
| `status` | `ok`, `parse_error`(예측 문장이 비었거나 Matcher 실패), `video_error`(영상 디코딩 실패). `ok`가 아니면 `prediction`은 `null`이다. |
| `prediction.answer` | 최종 선택지 문자. 선택지를 주지 않으면 `null` |
| `prediction.next_event` | 최종 예측 문장 |
| `prediction.option_probs` | 최종 답으로 쓰인 후보의 예측 문장에 대해 Matcher가 낸 선택지별 확률. 선택지를 주지 않으면 비어 있다. |
| `prediction.top_k` | 확률 상위 3개 선택지와 그 문장. 선택지를 주지 않으면 비어 있다. |
| `prediction.confidence` | 최종 답으로 쓰인 후보에서 Matcher가 `prediction.answer` 선택지에 준 확률. 선택지를 주지 않으면 `null` |
| `prediction.uncertain` | `confidence`가 `uncertain_tau`(기본 0.5)보다 낮으면 `true`. 선택지를 주지 않으면 `null` |
| `trace.nep_agent.final_source` | `prediction`이 어디에서 왔는지: `direct`(확신도가 임계값 이상이어서 Direct 답을 쓴 경우, 또는 설정이 `final_answer: direct`인 경우), 변형 이름(`tool_reason`, `causal_sft_nc`), `direct_fallback`(추론 후보가 모두 없어 Direct 사용) |
| `trace.nep_agent.reasoning` | 실행한 후보만 들어 있다. 확신도가 임계값 이상이어서 Direct 답으로 끝난 문항에서는 비어 있다. |
| `trace.nep_agent.reasoning.<변형>.record` | 추론 JSON을 검증한 결과: `terminal_state`, `ongoing_action`, `supporting_evidence`(근거 프레임 ID), `intermediate_states`, `causal_transition`, `predicted_event` |
| `trace.nep_agent.reasoning.<변형>.candidate` | 그 추론의 예측 문장(`event`)과 Matcher 결과(`choice`, `option_probs`). 추론 JSON이 실패하면 `null`이고 `raw_reply`에 모델 응답 전체가 남는다. |
| `trace.nep_agent.reasoning.tool_reason.selection` | 고른 도구(`tool`), 프레임 구간(`where`), 확인할 대상(`cue`), 이유(`reason`) |
| `trace.nep_agent.reasoning.tool_reason.observation` | 도구 실행 결과: `status`(`executed`, `no_new_evidence`, `failed:...`, `plan_failed`), 새로 생긴 근거 ID(`new_evidence_ids`). `audio_segment`가 실행되면 가져온 텍스트가 `evidence_text`에 남는다. 도구 선택 JSON이 무효이면 기본 계획(`selection.default_plan: true`)으로 대체되고, 기본 계획도 만들 수 없을 때만 `selection.raw`에 모델 응답 전체가 남는다. |
| `trace.direct.direct_prediction` | Direct 후보: 예측 문장(`event`), 선택지(`choice`), 확률(`option_probs`) |
| `trace.direct.routing` | Direct 문장의 생성 확신도(`generation_confidence` = exp(토큰 평균 로그확률)), 임계값(`gate_tau`), 추가 단계를 실행했는지(`routed`) |
| `trace.open.question_text` | 선택지 표현을 제거한 뒤 실제로 모델에 준 질문 |
| `trace.vlm_calls`, `trace.latency_sec` | VLM 호출 수와 문항당 소요 시간(배치 시간을 문항 수로 나눈 값) |
| `trace.frames_used`, `trace.extra_frames` | 프롬프트에 들어간 이미지 수, 도구가 추가한 이미지 수 |
| `frames[]` | 사용한 프레임 목록: `frame_id`, `t_sec`, `source`(`uniform`, `dense_resample`, `terminal_state`, `zoom`), `path` |
| `run_config` | 사용한 설정 파일 경로 |

`analysis.event_chain`은 NEPAgent 경로에서 쓰지 않아 빈 목록이다.

정답과 비교하는 스크립트는 저장된 결과와 정답 파일만 읽으며 모델을 호출하지 않는다. 결과 위치는 환경 변수 `RESULTS_ROOT`(기본 `results`), 정답 파일 위치는 `LABELS_DIR`(기본 `data/labels`)로 지정한다(`scripts/eval_common.py`). 표는 `results/quantitative/tables/`에 저장한다.

```bash
# 표 1: 순수 Qwen 대비 NEPAgent (정확도와 차이)
PYTHONPATH=. python scripts/main_table.py --phase test --block main --audio-types Speech --out results/quantitative/tables/main_test.md

# 정책 비교: 모든 후보를 저장한 실행에서 정책만 바꿈 (Direct만 / 인과 추론 항상 / 도구 + 추론 항상 / 임계값)
PYTHONPATH=. python scripts/main_table.py --phase test --block policy --audio-types Speech --out results/quantitative/tables/policy_test.md

# 표 2·3: 도구 메뉴 ablation (도구 1종 강제 실행과 최종 구성)
PYTHONPATH=. python scripts/tool_ablation.py --out results/quantitative/tables/tool_ablation_test.md

# 유형별 표: FutureBench는 hop별, FutureOmni는 질문 유형별
PYTHONPATH=. python scripts/subtype_table.py --phase test --audio-types Speech --out results/quantitative/tables/ablation_by_type_test.md
```

- `--phase test`는 `test`, `fb_test` split의 결과를 읽는다. `--audio-types Speech`는 FutureOmni에서 오디오 종류가 Speech인 문항만 남긴다(FutureBench는 거르지 않음). 보고서의 평가 범위(FutureBench 200 + FutureOmni Speech 128)가 이 설정이다.
- 표의 행은 `configs/report_tables.yaml`에 정의한다. 행의 종류(`kind`)는 `run`(한 실행에 저장된 `prediction.answer`)과 `tool`(모든 후보를 저장한 실행에 정책 `direct`, `reason`, `tool`, `gate@<임계값>`을 적용한 답)이다.
- 결과가 없는 행은 `Pending`으로 나온다.

### 결과

<!-- RESULTS -->

NEPAgent(`configs/nep_agent.yaml`)와 순수 Qwen3-VL-8B(`configs/plain_qwen.yaml`)의 test 정확도(%)다. FutureOmni는 Speech 문항만 집계한다.

| 범위 | 문항 | 순수 Qwen | NEPAgent | 차이 |
|---|---:|---:|---:|---:|
| 전체 | 328 | 44.8 | **50.6** | +5.8 |
| FutureBench | 200 | 50.0 | 57.5 | +7.5 |
| FutureOmni Speech | 128 | 36.7 | 39.8 | +3.1 |

도구 메뉴 ablation(test, %):

| 구성 | 전체 328 | FutureBench 200 | FutureOmni Speech 128 | 라우팅된 150문항 |
|---|---:|---:|---:|---:|
| 순수 Qwen | 44.8 | 50.0 | 36.7 | 41.3 |
| 마지막 상태 재확인만 (1종 강제) | 50.0 | 56.5 | 39.8 | 48.0 |
| 구간 재샘플링만 (1종 강제) | 50.0 | 57.0 | 39.1 | 48.0 |
| 영역 확대만 (1종 강제) | 48.8 | 57.0 | 35.9 | 45.3 |
| 도구 3종 중 모델 선택 (NEPAgent) | **50.6** | **57.5** | **39.8** | **49.3** |

- NEPAgent는 test 328문항에서 순수 Qwen보다 5.8pp 높다(FutureBench +7.5pp, FutureOmni Speech +3.1pp).
- 확신도가 임계값(τ = 0.75) 미만인 문항은 150/328(46%)이다. 이 문항에서 모델이 고른 도구는 마지막 상태 재확인 113회, 구간 재샘플링 37회, 영역 확대 0회다.
- 위 표는 4.2의 벤치마크 실행 뒤 `scripts/main_table.py`, `scripts/tool_ablation.py`로 다시 만들 수 있다. 문항별 실행 결과는 저장소에 넣지 않았다. 정성 사례, 실패 분석, 한계는 `docs/Technical_Report.md`에 있다.

### 테스트 케이스 결과물

- `results/qualitative/`: 조건이 다른 테스트 케이스 3건(도구별 1건: 마지막 상태 재확인, 구간 재샘플링, 영역 확대). 케이스마다 폴더 하나에 관찰 영상, 질문과 보기, NEPAgent 출력 요약, 보고서 그림이 있다. 설명은 `results/qualitative/qualitative.md`에 있다.
- `results/failure_cases/`: 실패 사례 2건(확신도 게이트 통과 오답, 도구 사용 후 혼동). 구성은 위와 같고 설명은 `results/failure_cases/failure_cases.md`에 있다.
- 프레임 캐시는 `results/`에 들어 있지 않다. 결과 JSON의 `frames[].path`는 `cache/<sample_id>/...` 형태의 상대 경로다.

케이스 JSON 하나를 읽는 순서:

1. `trace.direct.direct_prediction.event`와 `trace.direct.routing.generation_confidence`: Direct 문장과 확신도. `routing.routed`가 `false`면 확신도가 임계값 이상이어서 여기서 끝난 것이다.
2. `trace.nep_agent.reasoning.tool_reason.selection`과 `.observation`: 고른 도구와 구간, 실행 결과와 새 근거 ID.
3. `trace.nep_agent.reasoning.<변형>.record`: 마지막 상태, 진행 중인 행동, 근거 프레임, 중간 상태, 인과 전이, 예측 사건. 추론이 실패했으면 `null`이다.
4. `trace.nep_agent.final_source`와 `prediction`: 최종 답이 어디에서 왔는지와 최종 예측 문장, 선택지, 확률.

## 7. 저장소 구성

| 경로 | 내용 |
|---|---|
| `configs/` | 실행 설정 YAML(4.2의 표)과 결과 표 정의 |
| `scripts/run_agent.py`, `scripts/run_benchmark.py` | 단일 영상 실행, 벤치마크 실행 |
| `scripts/download_data.sh`, `scripts/build_splits*.py`, `scripts/build_full_splits.py`, `scripts/prepare_prefix.py` | 데이터 다운로드, 질문/정답 파일 생성, `prefix.mp4` 준비 |
| `scripts/precompute_audio.py` | Whisper·PANNs 결과 캐시 생성 |
| `scripts/train_stage1.sh`, `scripts/drop_event_chain.py`, `scripts/merge_reasoning_sft.py`, `scripts/teacher_quality.py` | LoRA 학습 실행, 학습 데이터 가공, Teacher 데이터 점검 |
| `scripts/main_table.py`, `scripts/tool_ablation.py`, `scripts/subtype_table.py`, `scripts/policy_eval.py`, `scripts/build_routed_splits.py`, `scripts/eval_common.py` | 저장된 결과로 표를 만드는 평가 스크립트(정답 파일을 읽음) |
| `src/agent/` | 프레임 샘플링과 프레임 번호 그리기, 도구, 프롬프트 블록, 결과 기록, 요약 그림(`figure.py`), 배치 실행(`pipeline.py`) |
| `src/stages/` | Direct, Matcher, 도구 선택·실행, 질문 변환, 단계별 프롬프트(`prompts/`) |
| `src/nep_agent/` | 인과 추론 변형과 JSON 검증, 전체 흐름(`core.py`), 추론 프롬프트(`prompts/`) |
| `src/models/` | VLM 인터페이스(vLLM, 테스트용 Mock), 오디오 모델(Whisper, PANNs) |
| `src/data/` | FutureOmni·FutureBench 로더, 영상 자르기, `prefix.mp4` 준비 |
| `src/train/` | 학습 데이터 생성(`build_sft_nep.py`, `build_sft_reasoning.py`), 학습 드라이버(`run_sft.sh`), 어댑터 병합(`merge_adapter.py`), LLaMA-Factory 설정(`configs/`), Teacher 프롬프트(`prompts/`) |
| `src/eval/` | 정확도 등 평가 지표 계산 |
| `src/utils/` | JSON 캐시, ffmpeg 도우미 |
| `src/schemas.py` | 입력, 실행 설정(`RunConfig`), 문항별 상태 자료형 |
| `tests/` | 단위 테스트(Mock VLM 사용, GPU 불필요) |
| `docs/` | 제출 문서: `Executive_Summary.md`(핵심 요약), `Technical_Report.md`(상세 보고서), `System_Architecture.md`(구조와 단계별 입출력), `Usage_and_Implementation.md`(사용한 모델·데이터·OSS·AI 도구와 직접 구현 범위), `figures/`(그림) |
| `requirements.txt`, `requirements-train.txt` | 추론 환경 의존성, 학습·병합 환경(`.venv_train`) 의존성 |
| `.gitignore` | 저장소에서 제외하는 경로(`checkpoints/`, `cache/`, `data/`, `logs/`, `.env`) |
| `results/` | 테스트 케이스(`results/qualitative/`), 실패 사례(`results/failure_cases/`) (6절). 벤치마크를 실행하면 결과가 `results/quantitative/`에 생긴다 |

실행 중에 생기는 폴더: `data/`(다운로드, 질문·정답 파일, `prefix.mp4`), `cache/`(프레임·오디오 캐시, 학습 데이터), `checkpoints/`(어댑터와 병합 모델), `logs/`.

## 8. 학습 재현(선택)

학습 데이터는 V1-33K의 학습용 영상에서 만든다(`V1_33K_ROOT` 필요, 5.3). 평가 문항에 쓰인 영상은 `data/splits/questions_fb_{dev,test,full}.jsonl`을 읽어 제외하므로, 5.3과 5.4의 FutureBench split을 먼저 만들어 둔다. 이 파일 중 하나라도 없으면 `src/train/build_sft_nep.py`가 `FileNotFoundError`로 멈춘다. Charades와 Ego4D 출처는 항상 제외하며 이를 끄는 옵션은 없다.

학습과 병합은 가상환경 `.venv_train`에서 한다. `src/train/run_sft.sh`가 `.venv_train/bin/llamafactory-cli`와 `.venv_train/bin/python`을 직접 부른다.

```bash
python3.11 -m venv .venv_train
.venv_train/bin/pip install --upgrade pip
.venv_train/bin/pip install -r requirements-train.txt
```

`requirements-train.txt`는 `torch==2.8.0`, `transformers==4.57.1`, `peft==0.18.1`과 LLaMA-Factory를 설치한다. LLaMA-Factory는 0.9.5.dev0이며 PyPI에 없는 개발 버전이라 GitHub 저장소(`hiyouga/LLaMA-Factory`)에서 설치한다. 커밋을 고정하지 않았으므로 설치 시점의 저장소 상태를 받는다. 학습 설정은 이 버전의 `qwen3_vl_nothink` 템플릿을 쓴다.

```bash
# 1) 정답 문장만 있는 데이터 (입력: 프레임 32장 + 질문, 목표: 이후 구간 캡션). CPU 작업.
#    기본값: --limit 3000 --keep-first 1500 --out-dir cache/sft/stage1_nep_sft
#    고정 seed로 섞은 순서의 앞 3,000개를 만들고, 그중 1,500개를 train.json에 쓴다.
PYTHONPATH=. python -m src.train.build_sft_nep

# 2) 2단계 Teacher 추론 데이터 (GPU, Teacher = $VLM_PATH의 Qwen3-VL-8B)
#    1단계: 정답을 보여주지 않고 관찰 사실 작성 / 2단계: 정답을 보여주고 영상 이후의 상태와 인과 전이 작성
#    기본값: 1)의 train.json 문항을 읽고 cache/sft/reasoning_sft에 쓴다.
PYTHONPATH=. python -m src.train.build_sft_reasoning

# 2-1) Teacher 필터에서 탈락한 만큼 채울 추가 후보 900개 (1)의 3,000개와 겹치지 않는 구간)
PYTHONPATH=. python -m src.train.build_sft_nep --out-dir cache/sft/stage1_nep_sft_extra \
  --offset 3000 --limit 900 --keep-first 0
PYTHONPATH=. python -m src.train.build_sft_reasoning \
  --source cache/sft/stage1_nep_sft_extra/records.jsonl \
  --ids-from cache/sft/stage1_nep_sft_extra/train.json \
  --out-dir cache/sft/reasoning_sft_extra

# 3) 1,500개로 합치기 (기본 데이터 전부 + 추가 데이터를 순서대로)
python scripts/merge_reasoning_sft.py cache/sft/reasoning_sft cache/sft/reasoning_sft_extra \
  cache/sft/reasoning_sft_1500 1500

# 4) 관찰 사건 사슬 제거 (프롬프트를 causal_sft_nc 스키마로 바꾸고 목표에서 observed.events 삭제)
PYTHONPATH=. python scripts/drop_event_chain.py cache/sft/reasoning_sft_1500 cache/sft/reasoning_sft_no_chain

# 5) 도구 프레임 입력 만들기 (CPU). 4)의 1,500개 중 약 절반의 입력에 추론 때와 같은 코드로 도구 프레임을 더한다
#    (마지막 상태 재확인 80%, 구간 재샘플링 20%, 고정 seed). 목표는 그대로 둔다. 출력: cache/sft/reasoning_sft_tool_inputs
PYTHONPATH=. python -m src.train.build_sft_tool_inputs --workers 16

# 6) LoRA 학습 + 7) 어댑터 병합 (학습이 끝나면 checkpoints/nep_agent_lora_merged로 자동 병합)
#    설정 src/train/configs/nep_agent_tool_evidence.yaml: 데이터 cache/sft/reasoning_sft_tool_inputs, 출력 checkpoints/nep_agent_lora
bash scripts/train_stage1.sh --gpu 2 --config src/train/configs/nep_agent_tool_evidence.yaml
```

참고:

- 2-1)과 3)의 `_extra`, `_1500` 폴더 이름은 예시다. 다른 이름을 써도 된다. 4)의 출력 폴더는 5)가 읽는 `cache/sft/reasoning_sft_no_chain`이어야 하고, 5)의 출력 폴더는 학습 설정의 `dataset_dir`과 같아야 한다. 3)과 4)는 순서를 바꿔도 결과가 같다.
- `scripts/train_stage1.sh`는 인자 없이 실행해도 NEPAgent 어댑터를 학습한다(기본 설정 `nep_agent_tool_evidence.yaml`). `--config`로 학습 설정을 바꿀 수 있다. `--dataset-dir`, `--output`을 주지 않으면 그 설정의 `dataset_dir`, `output_dir`을 쓴다. 로그는 `--log`를 주지 않으면 `logs/train_<설정 이름>.log`에 쓴다.
- `--gpu`는 GPU 번호다(기본 2, `1,2,3`처럼 여러 개 가능). GPU 0을 쓰려면 `ALLOW_GPU0=1`을 함께 준다. 예: `ALLOW_GPU0=1 bash scripts/train_stage1.sh --gpu 0`
- 학습 시간 상한은 20시간이다. 상한을 넘기면 어댑터 없이 멈추고, 샘플을 줄여 다시 돌리지 않는다. 상한은 `--max-hours`로 바꾼다. `--halve-on-timeout 1`을 주면 상한을 넘겼을 때 샘플을 절반으로 줄여 한 번 더 돌린다. 그 밖의 인자는 `--max-samples`, `--max-steps`, `--base`(기반 모델 디렉터리, 기본 `VLM_PATH`), `--no-merge`다.
- 자동 병합은 기반 모델을 로컬 디렉터리에서 읽으므로 `VLM_PATH`(또는 `--base`)가 로컬 디렉터리여야 한다. `--no-merge`로 학습만 한 경우에는 3.4의 병합 명령을 따로 실행한다.
- 학습 설정(`src/train/configs/nep_agent_tool_evidence.yaml`): LoRA rank 64, alpha 128, 학습률 1e-5, 3 epoch, cosine, 유효 배치 16, 비전 타워 고정.
- Teacher 데이터 점검(선택, GPU): `PYTHONPATH=. python scripts/teacher_quality.py --dir <teacher.jsonl이 있는 폴더> --out results/quantitative/tables/teacher_quality.md`

## 9. 테스트

```bash
PYTHONPATH=. python -m pytest -q
```

Mock VLM을 쓰므로 GPU와 모델 가중치 없이 실행된다. `V1_33K_ROOT`의 `futurebench.json`이나 `data/` 폴더가 필요한 테스트는 해당 파일이 없으면 skip된다.

## 10. 환경 변수

| 변수 | 기본값 | 읽는 곳 | 설명 |
|---|---|---|---|
| `VLM_PATH` | `Qwen/Qwen3-VL-8B-Instruct` | `src/models/vlm.py`, `src/train/run_sft.sh` | 기반 VLM의 로컬 디렉터리(또는 Hub 이름). 병합과 학습에서는 로컬 디렉터리여야 한다. |
| `V1_33K_ROOT` | `data/raw/v1_33k` | `scripts/build_splits_fb.py`, `scripts/build_full_splits.py`, `scripts/prepare_prefix.py`, `src/train/build_sft_nep.py` | V1-33K 배포본 폴더(FutureBench 문항·영상, 학습용 주석) |
| `FUTUREOMNI_ROOT` | `data/downloads/FutureOmni` | `scripts/download_data.sh` | FutureOmni 다운로드 위치 |
| `VLM_GPU_FRACTION` | `0.6` | `src/models/vlm.py` | vLLM `gpu_memory_utilization` |
| `ZOOM_DETECTOR` | 영역 확대 도구의 물체 탐지 모델 이름 또는 경로 | `IDEA-Research/grounding-dino-tiny` |
| `VLM_MAX_MODEL_LEN` | `32768` | `src/models/vlm.py` | vLLM `max_model_len` |
| `HF_CLI` | `.venv/bin/hf` | `scripts/download_data.sh` | `hf` 실행 파일 경로 |
| `ALLOW_GPU0` | `0` | `src/train/run_sft.sh` | `1`이면 학습에 GPU 0 사용을 허용 |
| `RESULTS_ROOT` | `results` | `scripts/eval_common.py` | 결과 표 스크립트가 읽는 실행 결과 위치(`<RESULTS_ROOT>/<split>/<run_name>/`) |
| `LABELS_DIR` | `data/labels` | `scripts/eval_common.py` | 결과 표 스크립트가 읽는 정답 파일(`labels_<split>.jsonl`) 위치 |
| `VP_STYLE` | `number` | `src/agent/vp.py` | `const`로 두면 모든 프레임에 같은 글자 `frame`을 그린다(프레임 번호 ablation용). |
| `DECORD_EOF_RETRY_MAX` | `40960` | `src/agent/observer.py` | decord의 영상 끝부분 읽기 재시도 한도. 값이 없을 때만 이 기본값으로 설정된다. |

`CUDA_VISIBLE_DEVICES`는 코드에서 읽지 않지만 추론에 쓸 GPU를 고르는 데 쓴다. 학습 스크립트는 `--gpu` 값으로 이 변수를 직접 설정한다.

## 참고 문헌

- [1] Q. Chen, J. Fu, C. Li, M. Zhang, S.-K. Ng, X. Qiu, "FutureOmni: Evaluating Future Forecasting from Omni-Modal Context for Multimodal LLMs", arXiv:2601.13836. https://arxiv.org/abs/2601.13836
- [2] H. Wang, H. Liu, X. Liu, C. Du, K. Kawaguchi, Y. Wang, T. Pang, "Fostering Video Reasoning via Next-Event Prediction" (NEP, FutureBench, V1-33K), arXiv:2505.22457. https://arxiv.org/abs/2505.22457
- [3] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, I. Sutskever, "Robust Speech Recognition via Large-Scale Weak Supervision" (Whisper), arXiv:2212.04356. https://arxiv.org/abs/2212.04356
- [4] Q. Kong, Y. Cao, T. Iqbal, Y. Wang, W. Wang, M. D. Plumbley, "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition", arXiv:1912.10211. https://arxiv.org/abs/1912.10211
- [5] S. Bai, Y. Cai, R. Chen, et al., "Qwen3-VL Technical Report", arXiv:2511.21631. https://arxiv.org/abs/2511.21631
