#!/usr/bin/env python
"""Web demo of NEPAgent: upload a video, watch each stage of the run, read the predicted next event.

The page calls the same pipeline as scripts/run_agent.py (configs/nep_agent.yaml). The model is loaded once at start.
Usage (from the repository root, after README section 4):
  pip install gradio
  PYTHONPATH=. python demo/app.py            # then open http://localhost:7860
  PYTHONPATH=. python demo/app.py --share    # temporary public link
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import gradio as gr

from scripts.run_agent import make_sample
from src.agent.config import load_config
from src.agent.figure import draw_timeline
from src.agent.output import write_result
from src.agent.pipeline import build_services, run_batch
from src.data.splitter import observed_clip
from src.models.audio import AudioAnalyzer
from src.models.vlm import DEFAULT_MODEL, VLLMQwen, resolve_model_path
from src.stages.questions import GENERIC_QUESTION

CONFIG = "configs/nep_agent.yaml"
OUTPUT_DIR = Path("results/demo")
TOOL_NAMES = {"terminal_state": "마지막 상태 재확인", "dense_resample": "구간 재샘플링", "zoom": "영역 확대"}
SOURCE_NAMES = {"direct": "초기 예측", "tool_reason": "도구 사용 뒤 인과 추론", "causal_sft_nc": "도구 없는 인과 추론",
                "direct_fallback": "초기 예측 (추가 단계가 답을 내지 못함)"}
SERVICES = None


def steps_markdown(result: dict) -> str:
    """The run as numbered stages, in the order the agent executed them."""
    trace = result["trace"]
    direct, agent = trace.get("direct") or {}, trace.get("nep_agent") or {}
    routing = direct.get("routing") or {}
    source = agent.get("final_source", "direct")
    entry = (agent.get("reasoning") or {}).get(source) or {}
    record = entry.get("record") or {}
    selection = entry.get("selection") or direct.get("tool_selection") or {}
    observation = entry.get("observation") or direct.get("tool_observation") or {}
    confidence, tau = routing.get("generation_confidence"), routing.get("gate_tau")
    lines = [f"### 1. 초기 예측\n{(direct.get('direct_prediction') or {}).get('event')}"]
    if confidence is not None:
        verdict = "임계값보다 낮아 도구와 인과 추론으로 진행" if routing.get("routed") else "임계값 이상이라 초기 예측으로 종료"
        lines.append(f"### 2. 확신도 게이트\n확신도 **{confidence:.3f}** (임계값 {tau:g}) → {verdict}")
    if selection.get("tool"):
        where = " ~ ".join(selection.get("where") or [])
        lines.append(f"### 3. 도구 선택\n**{TOOL_NAMES.get(selection['tool'], selection['tool'])}** · 구간 {where or '-'} · 확인할 대상: {selection.get('cue') or '-'}")
        added = ", ".join(observation.get("new_evidence_ids") or []) or "없음"
        lines.append(f"### 4. 도구 실행\n추가된 프레임: {added}")
    if record:
        lines.append("### 5. 인과 추론\n"
                     f"- **관찰 · 마지막 상태**: {record.get('terminal_state')}\n"
                     f"- **관찰 · 진행 중 행동**: {record.get('ongoing_action')}\n"
                     f"- **근거 프레임**: {', '.join(record.get('supporting_evidence') or []) or '-'}\n"
                     f"- **추론 · 인과 전이**: {record.get('causal_transition')}")
    prediction = result["prediction"]
    lines.append(f"### 최종 예측 ({SOURCE_NAMES.get(source, source)})\n**{prediction['next_event']}**")
    if prediction.get("answer"):
        lines.append(f"대응된 보기: **{prediction['answer']}**")
    lines.append(f"모델 호출 {trace.get('vlm_calls')}회 · {trace.get('latency_sec', 0):.1f}초")
    return "\n\n".join(lines)


def predict(video: str | None, t_split: float, question: str, options_text: str, tau: float):
    if not video:
        raise gr.Error("영상을 올려 주세요.")
    options = [line.strip() for line in options_text.splitlines() if line.strip()]
    if options and not 2 <= len(options) <= 8:
        raise gr.Error("보기는 2~8개(한 줄에 하나)이거나 비워 두어야 합니다.")
    sample_id = f"demo_{int(time.time())}"
    prefix = observed_clip(video, t_split or None, OUTPUT_DIR / "work" / sample_id)
    sample = make_sample(sample_id, str(prefix), question.strip() or GENERIC_QUESTION, options)
    cfg = load_config(CONFIG, {"gate_tau": float(tau)})
    result = run_batch([sample], cfg, SERVICES)[0]
    if not result.get("prediction"):
        raise gr.Error(f"예측을 만들지 못했습니다: {result.get('status')}")
    path = write_result(result, OUTPUT_DIR)
    figure = path.with_suffix(".png")
    draw_timeline(result, figure)
    gallery = [(str(Path(frame["path"]).resolve()), f"{frame['frame_id']} · {frame['t_sec']:.1f}s" + ("" if frame["source"] == "uniform" else " · 도구"))
               for frame in result["frames"]]
    return steps_markdown(result), gallery, str(figure), json.loads(path.read_text(encoding="utf-8"))


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="NEPAgent demo") as page:
        gr.Markdown("# NEPAgent: 다음 사건 예측 데모\n영상을 올리면 관찰 구간만 보고 직후에 일어날 사건을 예측합니다. 단계별 출력과 사용한 프레임을 함께 보여 줍니다.")
        with gr.Row():
            with gr.Column(scale=1):
                video = gr.Video(label="입력 영상 (관찰 구간)")
                t_split = gr.Number(label="관찰 구간 끝(초). 0이면 파일 전체", value=0)
                question = gr.Textbox(label="질문 (선택)", value=GENERIC_QUESTION)
                options = gr.Textbox(label="보기 (선택, 한 줄에 하나). 예측이 끝난 뒤 대응에만 사용", lines=4)
                tau = gr.Slider(0.0, 1.0, value=0.75, step=0.05, label="확신도 임계값 (제출 설정 0.75, 1.0이면 항상 도구와 인과 추론 실행)")
                run = gr.Button("예측 실행", variant="primary")
                gr.Examples([["demo/video.mp4", 0, GENERIC_QUESTION, "", 0.75]], [video, t_split, question, options, tau])
            with gr.Column(scale=1):
                steps = gr.Markdown(label="실행 과정")
        frames = gr.Gallery(label="모델이 본 프레임 (번호 표시, '도구'는 도구가 추가한 프레임)", columns=8, height=420)
        figure = gr.Image(label="요약 그림")
        raw = gr.JSON(label="결과 JSON 원본", open=False)
        run.click(predict, [video, t_split, question, options, tau], [steps, frames, figure, raw])
    return page


def main() -> None:
    global SERVICES
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="create a temporary public gradio.live link")
    args = parser.parse_args()
    cfg = load_config(CONFIG)
    audio = AudioAnalyzer() if cfg.use_audio else None
    SERVICES = build_services(VLLMQwen(resolve_model_path(DEFAULT_MODEL, cfg.lora_path)), "cache", audio)
    build_ui().queue(max_size=8).launch(server_name="0.0.0.0", server_port=args.port, share=args.share, show_error=True,
                                    allowed_paths=[str(Path("cache").resolve()), str(OUTPUT_DIR.resolve())])


if __name__ == "__main__":
    main()
