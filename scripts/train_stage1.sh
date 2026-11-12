#!/usr/bin/env bash
# LoRA SFT and adapter merge. Default: the submitted NEPAgent adapter (src/train/configs/nep_agent_tool_evidence.yaml).
# The dataset and output directories are read from the config unless --dataset-dir / --output are given.
#   scripts/train_stage1.sh --gpu 1                                                   # NEPAgent reasoning adapter
STAGE=1 DEFAULT_MAX_HOURS=20 DEFAULT_HALVE=0
source "$(dirname "${BASH_SOURCE[0]}")/../src/train/run_sft.sh"
