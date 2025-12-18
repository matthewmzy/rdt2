#!/bin/bash
# ============================================================
# RDT2 Flow Matching Action Expert 微调脚本 - Pika 数据集
# ============================================================
#
# 说明:
#   这个脚本用于训练 Flow Matching action expert (RDTRunner)
#   而不是 VQ tokenizer 模式
#
#   Flow Matching 的优势:
#   - 推理延迟更低 (5步去噪 vs autoregressive 27 tokens)
#   - 更平滑的动作输出
#   - 适合实时控制场景
#
# 架构:
#   RDT2-VQ (frozen VLM) -> RDTRunner (Flow Matching action expert, ~4M params)
#                           ^-- 这是我们要训练的部分
#
# 对比 VQ 模式:
#   - VQ 模式: 训练 7B VLM 的 LoRA，5000 步即可
#   - FM 模式: 全参数训练 4M 小模型，需要 10000+ 步
#   - FM 不支持 LoRA (模型太小，不需要)
#
# 用法:
#   bash scripts/finetune_pika_fm.sh
#
# ============================================================

set -e

# ===================== 配置区域 =====================

# 任务名称
TASK="pika-shoes-fm"

# 模型配置
CONFIG_PATH="configs/rdt/post_train.yaml"
VLM_MODEL="robotics-diffusion-transformer/RDT2-VQ"  # 冻结的 VLM backbone
# 可选: 从预训练的 FM 模型开始
PRETRAINED_FM="robotics-diffusion-transformer/RDT2-FM"
# PRETRAINED_FM=""  # 留空则从头训练 RDTRunner

# 数据配置
DATASET_CONFIG="rdt2_pika_1212_pick_up_shoes/dataset_config.yaml"

# 输出目录
OUTPUT_DIR="./outputs/rdt2-fm-${TASK}"
LOGGING_DIR="./logs/rdt2-fm-${TASK}"

# 训练超参数
# 注意: FM 模式训练的是 RDTRunner (~4M 参数)，不是 7B VLM
# 由于模型较小且全参数训练，需要比 VQ LoRA 模式更多的步数
TRAIN_BATCH_SIZE=48       # Flow Matching 可以用更大的 batch size (模型小)
SAMPLE_BATCH_SIZE=16      # 用于采样验证
GRADIENT_ACCUMULATION_STEPS=4  # 有效 batch = 32 * 4 = 128
MAX_TRAIN_STEPS=10000     # 从预训练 FM 微调，10k 步应该够了
LEARNING_RATE=1e-4
LR_WARMUP_STEPS=500
CHECKPOINTING_PERIOD=1000
SAMPLE_PERIOD=500         # 每 500 步采样一次验证
CHECKPOINTS_TOTAL_LIMIT=10
DATALOADER_NUM_WORKERS=8

# HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com

# 确保使用 conda 环境的库
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export LD_PRELOAD="$CONDA_PREFIX/lib/libjpeg.so.8"

# 分布式训练环境变量
export MASTER_ADDR=localhost
export MASTER_PORT=29501
export WORLD_SIZE=2
export RANK=0
export LOCAL_RANK=0

# CUDA 显存优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ===================== 辅助函数 =====================

print_header() {
    echo ""
    echo "============================================================"
    echo "$1"
    echo "============================================================"
}

check_conda() {
    if ! command -v conda &> /dev/null; then
        echo "Error: conda not found."
        exit 1
    fi
}

activate_conda() {
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate rdt2
    echo "Activated: rdt2"
}

# ===================== 训练 =====================

train_fm() {
    print_header "开始 Flow Matching Action Expert 训练"
    
    echo "任务: $TASK"
    echo "VLM Backbone: $VLM_MODEL"
    echo "配置文件: $CONFIG_PATH"
    echo "数据集: $DATASET_CONFIG"
    echo "输出目录: $OUTPUT_DIR"
    echo "Batch Size: $TRAIN_BATCH_SIZE"
    echo "Max Steps: $MAX_TRAIN_STEPS"
    echo ""
    
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$LOGGING_DIR"
    
    # 构建命令
    CMD="PYTHONPATH=$(pwd) accelerate launch rdt/main.py \
        --deepspeed=scripts/zero1.json \
        --config_path=$CONFIG_PATH \
        --pretrained_vision_language_model_name_or_path=$VLM_MODEL \
        --webdataset_config=$DATASET_CONFIG \
        --output_dir=$OUTPUT_DIR \
        --train_batch_size=$TRAIN_BATCH_SIZE \
        --sample_batch_size=$SAMPLE_BATCH_SIZE \
        --gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
        --max_train_steps=$MAX_TRAIN_STEPS \
        --learning_rate=$LEARNING_RATE \
        --lr_scheduler=cosine \
        --lr_warmup_steps=$LR_WARMUP_STEPS \
        --checkpointing_period=$CHECKPOINTING_PERIOD \
        --sample_period=$SAMPLE_PERIOD \
        --checkpoints_total_limit=$CHECKPOINTS_TOTAL_LIMIT \
        --dataloader_num_workers=$DATALOADER_NUM_WORKERS \
        --mixed_precision=bf16 \
        --image_aug \
        --report_to=wandb \
        --logging_dir=$LOGGING_DIR"
    
    # 如果有预训练的 FM 模型，添加参数
    if [ -n "$PRETRAINED_FM" ]; then
        CMD="$CMD --pretrained_model_name_or_path=$PRETRAINED_FM"
        echo "从预训练 FM 模型继续: $PRETRAINED_FM"
    else
        echo "从头训练 RDTRunner"
    fi
    
    echo ""
    echo "运行命令:"
    echo "$CMD"
    echo ""
    
    # 执行
    eval $CMD
    
    print_header "训练完成!"
    echo "模型保存在: $OUTPUT_DIR"
}

# ===================== 主流程 =====================

main() {
    cd "$(dirname "$0")/.."
    
    check_conda
    activate_conda
    
    case "${1:-}" in
        --help|-h)
            echo "用法: bash scripts/finetune_pika_fm.sh [选项]"
            echo ""
            echo "选项:"
            echo "  --help, -h    显示帮助"
            echo ""
            echo "Flow Matching vs VQ 模式:"
            echo "  - VQ 模式 (finetune_pika.sh): 训练整个 VLA，action 用 VQ tokenizer"
            echo "  - FM 模式 (本脚本): 训练 Flow Matching action expert，VLM 冻结"
            echo ""
            echo "推荐工作流:"
            echo "  1. 先用 VQ 模式微调 VLA (finetune_pika.sh)"
            echo "  2. 再用 FM 模式训练 action expert 以获得更低延迟"
            ;;
        *)
            train_fm
            ;;
    esac
}

main "$@"
