#!/usr/bin/env bash
# Llama-3.2-3B-Instruct version of train_sft_book_coldstart.sh
set -euo pipefail
set -x

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
CONDA_ENV_ROOT="/opt/anaconda3/envs/linh_recr1"
TORCHRUN_BIN="$CONDA_ENV_ROOT/bin/torchrun"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
MAX_LENGTH="${MAX_LENGTH:-2560}"

"$TORCHRUN_BIN" --standalone --nnodes=1 --nproc_per_node=1 \
  -m verl.trainer.fsdp_sft_trainer \
  data.train_files=/data/uqlinh/Reason4Rec/Data/Book_data/recone_sft/train.parquet \
  data.val_files=/data/uqlinh/Reason4Rec/Data/Book_data/recone_sft/val.parquet \
  data.prompt_key=prompt \
  data.response_key=response \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.max_length="$MAX_LENGTH" \
  data.truncation=error \
  data.micro_batch_size="$MICRO_BATCH_SIZE" \
  model.partial_pretrain=meta-llama/Llama-3.2-3B-Instruct \
  model.enable_gradient_checkpointing=True \
  model.fsdp_config.cpu_offload=True \
  model.fsdp_config.offload_params=True \
  trainer.default_local_dir=/data/uqlinh/reczero/checkpoints/yelp/sft_coldstart_llama \
  trainer.project_name=recone-sft \
  trainer.experiment_name=book-coldstart-llama-3B \
  trainer.total_epochs=3 \
  trainer.logger=['console','wandb'] \
  "$@"
