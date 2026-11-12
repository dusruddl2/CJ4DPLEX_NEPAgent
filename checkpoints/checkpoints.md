# checkpoints/

직접 학습한 가중치는 저장소에 넣지 않았습니다(과제 안내 2.4절). 아래 Google Drive 폴더에서 받아 이 폴더에 풀어 주세요.

**다운로드**: https://drive.google.com/drive/folders/1GoackN2mI0RrBRY4ba3xzBhlcIUMVR1i?usp=sharing

| 파일 | 내용 | 압축 해제 위치 |
|---|---|---|
| `nep_agent_lora.zip` (약 620MB) | NEPAgent LoRA 어댑터. Qwen3-VL-8B-Instruct 기반, r 64 / α 128, 1,500 클립 × 3 epoch | `checkpoints/nep_agent_lora/` |
| `nep_agent_lora_merged.zip` | 위 어댑터를 기반 모델에 병합한 모델. 실행 시 실제로 로드하는 모델 | `checkpoints/nep_agent_lora_merged/` |

## 사용 방법

**방법 A. 병합 모델을 받는 경우 (권장)**

```bash
cd checkpoints
unzip nep_agent_lora_merged.zip   # → checkpoints/nep_agent_lora_merged/
```

받은 뒤 바로 실행할 수 있습니다.

**방법 B. 어댑터만 받는 경우**

```bash
cd checkpoints
unzip nep_agent_lora.zip          # → checkpoints/nep_agent_lora/
```

어댑터를 기반 모델에 병합해야 실행할 수 있습니다. 병합 명령은 저장소 루트 `README.md` 3.4절에 있습니다.

## 확인

실행 설정(`configs/nep_agent.yaml`)은 `checkpoints/nep_agent_lora_merged/`를 로드합니다. 이 폴더가 없으면 실행이 `FileNotFoundError`로 멈춥니다.
