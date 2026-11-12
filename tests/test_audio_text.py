"""The final configuration shows the speech transcript and no sound-tag lines."""
from dataclasses import replace

from src.agent.config import load_config
from src.schemas import AgentInput, SampleState
from src.stages.direct import common
from src.stages.state import DirectState


def _state():
    frames = tuple({"frame_id": f"#{i:02d}", "t_sec": float(i), "source": "uniform", "path": "x.jpg", "raw_path": "x.jpg", "caption": ""} for i in (1, 2))
    base = SampleState(input=AgentInput("s", "p.mp4", "q", ()), frames=frames)
    state = DirectState(**{f: getattr(base, f) for f in base.__dataclass_fields__})
    return replace(state, question_text="q", audio={"asr": [{"t_start": 1.0, "t_end": 2.0, "text": "hello there"}],
                                                   "tags": [{"t_sec": 1.0, "t_end": 2.0, "label": "Music", "score": 0.9}]})


def test_final_config_keeps_speech_and_drops_sound_tags():
    cfg = load_config("configs/nep_agent.yaml")
    assert cfg.audio_tags is False
    text = common(_state(), cfg)["audio"]
    assert "hello there" in text and "sound:" not in text


def test_tags_shown_when_enabled():
    cfg = replace(load_config("configs/nep_agent.yaml"), audio_tags=True)
    assert "sound: Music" in common(_state(), cfg)["audio"]
