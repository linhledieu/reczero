
export RAY_TMPDIR=/data/uqlinh/reczero/cache/ray

# System temp dirs (Ray, HF, Python all respect this)
export TMPDIR=/data/uqlinh/reczero/cache/tmp
export TMP=/data/uqlinh/reczero/cache/tmp
export TEMP=/data/uqlinh/reczero/cache/tmp

mkdir -p $RAY_TMPDIR $TMPDIR

export HF_HOME=/data/uqlinh/reczero/cache/hf
export TRANSFORMERS_CACHE=/data/uqlinh/reczero/cache/hf/transformers
export HF_DATASETS_CACHE=/data/uqlinh/reczero/cache/hf/datasets
export HUGGINGFACE_HUB_CACHE=/data/uqlinh/reczero/cache/hf/hub



#!/usr/bin/env bash
# Llama-3.2-3B-Instruct version of main_grpo_yelp.sh
set -euo pipefail
set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


CKPT_ROOT="/data/uqlinh/reczero/checkpoints/yelp_llama"
ACTOR_DIR="$CKPT_ROOT/actor"
BASE_MODEL="meta-llama/Llama-3.2-3B-Instruct"

SLEEP_SEC=30
MAX_RETRIES=5
MIN_TRAIN_BSZ=2

TRAIN_BSZ=2
VAL_BSZ=3
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-6000}
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-2048}
ROLLOUT_N=4
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-1.0}
ROLLOUT_TOP_K=${ROLLOUT_TOP_K:--1}
ROLLOUT_REPETITION_PENALTY=${ROLLOUT_REPETITION_PENALTY:-1.0}
ROLLOUT_GPU_MEM_UTIL=0.5
PPO_MINI_BSZ=16
PPO_MICRO_BSZ=4
LOGPROB_MICRO_BSZ=16
USE_DYNAMIC_BSZ=True
ACTOR_MAX_TOKENS_PER_GPU=16384
CRITIC_MAX_TOKENS_PER_GPU=32768
ROLLOUT_TP_SIZE=1
ACTOR_LR=3e-7
USE_KL_LOSS=true
ACTOR_KL_LOSS_COEF=0.001
ALG_KL_COEF=0.001
SAVE_FREQ=100
TEST_FREQ=2
TOTAL_EPOCHS=1
EARLY_STOP_MAE=${EARLY_STOP_MAE:-0.99}
RESUME_FROM_LATEST=${RESUME_FROM_LATEST:-0}
PINNED_START_CKPT="/data/uqlinh/reczero/checkpoints/yelp/sft_coldstart_llama/global_step_1059"

latest_ckpt () {
  ls -1d "$ACTOR_DIR"/global_step_* 2>/dev/null \
    | sed -E 's|.*/global_step_([0-9]+)$|\1\t&|' \
    | sort -n \
    | tail -n 1 \
    | cut -f2-
}

CHECK_GPUS=(1 2)
CHECK_INTERVAL=300
MIN_FREE_MEM_MIB=20000

gpu_has_required_free_mem () {
  local gpu="$1"
  local free_mem
  free_mem="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | awk '{gsub(/ /, ""); print $1}' || true)"

  [[ -z "$free_mem" ]] && return 1
  [[ "$free_mem" =~ ^[0-9]+$ ]] || return 1

  (( free_mem >= MIN_FREE_MEM_MIB ))
}

wait_for_gpus_ready () {
  while true; do
    local all_ready=1
    for g in "${CHECK_GPUS[@]}"; do
      local free_mem
      free_mem="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" 2>/dev/null | awk '{gsub(/ /, ""); print $1}' || true)"

      if gpu_has_required_free_mem "$g"; then
        echo "[gpu-check] GPU $g free memory ${free_mem} MiB >= ${MIN_FREE_MEM_MIB} MiB"
      else
        echo "[gpu-check] GPU $g free memory ${free_mem:-unknown} MiB < ${MIN_FREE_MEM_MIB} MiB"
        all_ready=0
      fi
    done

    if (( all_ready == 1 )); then
      echo "[gpu-check] GPUs ${CHECK_GPUS[*]} are ready. Starting run."
      return 0
    fi

    echo "[gpu-check] Not enough free memory yet. Rechecking in ${CHECK_INTERVAL}s (5 min)..."
    sleep "$CHECK_INTERVAL"
  done
}

attempt=0
while (( attempt < MAX_RETRIES )); do
  attempt=$((attempt + 1))
  wait_for_gpus_ready

  MODEL_PATH=""
  if [[ "$RESUME_FROM_LATEST" == "1" ]]; then
    MODEL_PATH="$(latest_ckpt || true)"
    if [[ -n "${MODEL_PATH}" ]]; then
      echo "[run $attempt] Resuming from latest checkpoint: $MODEL_PATH"
    fi
  elif [[ -d "$PINNED_START_CKPT" ]]; then
    MODEL_PATH="$PINNED_START_CKPT"
    echo "[run $attempt] Starting from pinned checkpoint: $MODEL_PATH"
  else
    MODEL_PATH="$(latest_ckpt || true)"
    if [[ -n "${MODEL_PATH}" ]]; then
      echo "[run $attempt] Pinned checkpoint not found. Falling back to latest checkpoint: $MODEL_PATH"
    fi
  fi

  if [[ -z "${MODEL_PATH}" ]]; then
    echo "[run $attempt] No checkpoint found in $ACTOR_DIR. Starting from base model: $BASE_MODEL"
    MODEL_PATH="$BASE_MODEL"
  fi


  LOG="grpo_llama_attempt_${attempt}_bsz${TRAIN_BSZ}_$(date +%Y%m%d_%H%M%S).log"
  echo "[run $attempt] Resuming from: $MODEL_PATH"
  echo "[run $attempt] Using train_batch_size=$TRAIN_BSZ, val_batch_size=$VAL_BSZ, ppo_mini_batch_size=$PPO_MINI_BSZ, ppo_micro_batch_size=$PPO_MICRO_BSZ, log_prob_micro_batch_size=$LOGPROB_MICRO_BSZ"
  echo "[run $attempt] Rollout decode: max_prompt_len=$MAX_PROMPT_LEN, max_response_len=$MAX_RESPONSE_LEN, n=$ROLLOUT_N, temp=$ROLLOUT_TEMPERATURE, top_p=$ROLLOUT_TOP_P, top_k=$ROLLOUT_TOP_K, repetition_penalty=$ROLLOUT_REPETITION_PENALTY"

  set +e
  python3 -m verl.trainer.main_ppo_llama \
    algorithm.adv_estimator=grpo \
    data.train_files=/data/uqlinh/reczero/data/yelp_llama/train.parquet \
    data.val_files=/data/uqlinh/reczero/data/yelp_llama/test.parquet \
    data.train_batch_size="$TRAIN_BSZ" \
    data.val_batch_size="$VAL_BSZ" \
    data.max_prompt_length="$MAX_PROMPT_LEN" \
    data.max_response_length="$MAX_RESPONSE_LEN" \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=$ACTOR_LR \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BSZ" \
    actor_rollout_ref.actor.ppo_micro_batch_size="$PPO_MICRO_BSZ" \
    actor_rollout_ref.actor.use_dynamic_bsz="$USE_DYNAMIC_BSZ" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$ACTOR_MAX_TOKENS_PER_GPU" \
    actor_rollout_ref.actor.use_kl_loss=$USE_KL_LOSS \
    actor_rollout_ref.actor.kl_loss_coef=$ACTOR_KL_LOSS_COEF \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size="$LOGPROB_MICRO_BSZ" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP_SIZE" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEM_UTIL" \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    actor_rollout_ref.ref.log_prob_micro_batch_size="$LOGPROB_MICRO_BSZ" \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz="$USE_DYNAMIC_BSZ" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$ACTOR_MAX_TOKENS_PER_GPU" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    algorithm.kl_ctrl.kl_coef=$ALG_KL_COEF \
    trainer.critic_warmup=0 \
    critic.use_dynamic_bsz="$USE_DYNAMIC_BSZ" \
    critic.ppo_max_token_len_per_gpu="$CRITIC_MAX_TOKENS_PER_GPU" \
    critic.forward_max_token_len_per_gpu="$CRITIC_MAX_TOKENS_PER_GPU" \
    trainer.logger='['wandb']' \
    trainer.project_name='GRPO_logic_KK' \
    trainer.experiment_name='Llama-3.2-3B' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.default_local_dir="$CKPT_ROOT" \
    trainer.default_hdfs_dir=null \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.test_freq="$TEST_FREQ" \
    +trainer.save_total_limit=2 \
    trainer.total_epochs="$TOTAL_EPOCHS" \
    actor_rollout_ref.rollout.temperature=$ROLLOUT_TEMPERATURE \
    actor_rollout_ref.rollout.top_p=$ROLLOUT_TOP_P \
    actor_rollout_ref.rollout.top_k=$ROLLOUT_TOP_K \
    +actor_rollout_ref.rollout.repetition_penalty=$ROLLOUT_REPETITION_PENALTY \
    ${EARLY_STOP_MAE:++trainer.early_stop_mae=$EARLY_STOP_MAE} \
    "$@" 2>&1 | stdbuf -oL -eL tee "$LOG"

  rc=${PIPESTATUS[0]}
  set -e

  if [[ $rc -eq 0 ]]; then
    echo "[run $attempt] Finished successfully."
    exit 0
  fi

  if (( TRAIN_BSZ > MIN_TRAIN_BSZ )); then
    NEW_BSZ=$(( TRAIN_BSZ / 2 ))
    (( NEW_BSZ < MIN_TRAIN_BSZ )) && NEW_BSZ=$MIN_TRAIN_BSZ
    VAL_BSZ=$(( VAL_BSZ > 1 ? VAL_BSZ / 2 : 1 ))
    PPO_MINI_BSZ=$(( PPO_MINI_BSZ > 1 ? PPO_MINI_BSZ / 2 : 1 ))
    PPO_MICRO_BSZ=$(( PPO_MICRO_BSZ > 1 ? PPO_MICRO_BSZ / 2 : 1 ))
    LOGPROB_MICRO_BSZ=$(( LOGPROB_MICRO_BSZ > 1 ? LOGPROB_MICRO_BSZ / 2 : 1 ))
    echo "[run $attempt] Failed. Reducing batch sizes: train $TRAIN_BSZ → $NEW_BSZ, val -> $VAL_BSZ, ppo_mini -> $PPO_MINI_BSZ, ppo_micro -> $PPO_MICRO_BSZ, log_prob_micro -> $LOGPROB_MICRO_BSZ"
    TRAIN_BSZ=$NEW_BSZ
  else
    echo "[run $attempt] Failed and already at minimum train_batch_size=$MIN_TRAIN_BSZ"
    echo "[run $attempt] Giving up."
    exit 1
  fi

  echo "[run $attempt] Retrying in ${SLEEP_SEC}s..."
  sleep "$SLEEP_SEC"
done
