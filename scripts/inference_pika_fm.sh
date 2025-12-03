#!/bin/bash
# ============================================================
# RDT2 Flow Matching 离线推理脚本
# ============================================================
# 
# 用法:
#   1. 使用最新 checkpoint 推理:
#      bash scripts/inference_pika_fm.sh
#   
#   2. 指定 checkpoint 和样本:
#      bash scripts/inference_pika_fm.sh --checkpoint 10000 --shard 5 --sample 10
#   
#   3. 批量推理多个样本:
#      bash scripts/inference_pika_fm.sh --batch --num-samples 10
#
# ============================================================

set -e

# ===================== 配置区域 =====================

# 模型配置
VLM_MODEL="robotics-diffusion-transformer/RDT2-VQ"
FM_OUTPUT_DIR="./outputs/rdt2-fm-pika-bottle-fm"
CONFIG_PATH="configs/rdt/post_train.yaml"

# 数据配置
SHARD_DIR="rdt2_pika_shards"
NORMALIZER_PATH="normalizer.pt"

# 推理输出目录
INFERENCE_OUTPUT_DIR="inference_outputs_fm"

# HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com

# 确保使用conda环境的库
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export LD_PRELOAD="$CONDA_PREFIX/lib/libjpeg.so.8"

# ===================== 辅助函数 =====================

print_header() {
    echo ""
    echo "============================================================"
    echo "$1"
    echo "============================================================"
}

check_conda() {
    if ! command -v conda &> /dev/null; then
        echo "Error: conda not found. Please install Anaconda/Miniconda."
        exit 1
    fi
}

activate_conda() {
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate rdt2
    echo "Activated conda environment: rdt2"
}

get_latest_checkpoint() {
    # 获取最新的 checkpoint 目录
    latest=$(ls -d ${FM_OUTPUT_DIR}/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
    if [ -z "$latest" ]; then
        echo "Error: No checkpoint found in ${FM_OUTPUT_DIR}"
        exit 1
    fi
    echo "$latest"
}

# ===================== 推理函数 =====================

run_single_inference() {
    local checkpoint_path="$1"
    local shard_idx="$2"
    local sample_idx="$3"
    
    print_header "Running FM Single Inference"
    echo "FM Checkpoint: $checkpoint_path"
    echo "VLM Model: $VLM_MODEL"
    echo "Shard: $shard_idx"
    echo "Sample: $sample_idx"
    echo ""
    
    python pika_test_scripts/inference_offline_fm.py \
        --fm-checkpoint "$checkpoint_path" \
        --vlm-model "$VLM_MODEL" \
        --config-path "$CONFIG_PATH" \
        --normalizer-path "$NORMALIZER_PATH" \
        --shard-dir "$SHARD_DIR" \
        --shard-idx "$shard_idx" \
        --sample-idx "$sample_idx" \
        --output-dir "$INFERENCE_OUTPUT_DIR"
}

run_batch_inference() {
    local checkpoint_path="$1"
    local num_samples="${2:-10}"
    
    print_header "Running FM Batch Inference"
    echo "FM Checkpoint: $checkpoint_path"
    echo "VLM Model: $VLM_MODEL"
    echo "Number of samples: $num_samples"
    echo ""
    
    python pika_test_scripts/inference_offline_fm.py \
        --fm-checkpoint "$checkpoint_path" \
        --vlm-model "$VLM_MODEL" \
        --config-path "$CONFIG_PATH" \
        --normalizer-path "$NORMALIZER_PATH" \
        --shard-dir "$SHARD_DIR" \
        --output-dir "$INFERENCE_OUTPUT_DIR" \
        --batch \
        --num-samples "$num_samples"
    
    print_header "FM Batch Inference Complete"
    echo "Results saved in: $INFERENCE_OUTPUT_DIR"
}

# ===================== 使用说明 =====================

show_usage() {
    echo "用法: bash scripts/inference_pika_fm.sh [选项]"
    echo ""
    echo "选项:"
    echo "  --checkpoint <step>   指定 checkpoint step (默认: 最新)"
    echo "  --shard <idx>         指定 shard 索引 (默认: 0)"
    echo "  --sample <idx>        指定样本索引 (默认: 0)"
    echo "  --batch               批量推理模式"
    echo "  --num-samples <n>     批量推理的样本数量 (默认: 10)"
    echo "  --help                显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  bash scripts/inference_pika_fm.sh                           # 使用最新 checkpoint"
    echo "  bash scripts/inference_pika_fm.sh --checkpoint 10000        # 使用 checkpoint-10000"
    echo "  bash scripts/inference_pika_fm.sh --shard 5 --sample 10     # 指定 shard 和样本"
    echo "  bash scripts/inference_pika_fm.sh --batch --num-samples 10  # 批量推理 10 个样本"
}

# ===================== 主流程 =====================

main() {
    cd "$(dirname "$0")/.."  # 切换到项目根目录
    
    # 默认参数
    CHECKPOINT_STEP=""
    SHARD_IDX=0
    SAMPLE_IDX=0
    BATCH_MODE=false
    NUM_SAMPLES=10
    
    # 解析参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            --checkpoint)
                CHECKPOINT_STEP="$2"
                shift 2
                ;;
            --shard)
                SHARD_IDX="$2"
                shift 2
                ;;
            --sample)
                SAMPLE_IDX="$2"
                shift 2
                ;;
            --batch)
                BATCH_MODE=true
                shift
                ;;
            --num-samples)
                NUM_SAMPLES="$2"
                shift 2
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    check_conda
    activate_conda
    
    # 确定 checkpoint 路径
    if [ -z "$CHECKPOINT_STEP" ]; then
        CHECKPOINT_PATH=$(get_latest_checkpoint)
    else
        CHECKPOINT_PATH="${FM_OUTPUT_DIR}/checkpoint-${CHECKPOINT_STEP}"
        if [ ! -d "$CHECKPOINT_PATH" ]; then
            echo "Error: Checkpoint not found: $CHECKPOINT_PATH"
            echo "Available checkpoints:"
            ls -d ${FM_OUTPUT_DIR}/checkpoint-* 2>/dev/null || echo "  (none)"
            exit 1
        fi
    fi
    
    echo "Using FM checkpoint: $CHECKPOINT_PATH"
    
    # 创建输出目录
    mkdir -p "$INFERENCE_OUTPUT_DIR"
    
    if [ "$BATCH_MODE" = true ]; then
        run_batch_inference "$CHECKPOINT_PATH" "$NUM_SAMPLES"
    else
        run_single_inference "$CHECKPOINT_PATH" "$SHARD_IDX" "$SAMPLE_IDX"
    fi
    
    print_header "Done!"
    echo "输出目录: $INFERENCE_OUTPUT_DIR"
    echo ""
    echo "查看结果:"
    echo "  - 可视化图片: ${INFERENCE_OUTPUT_DIR}/fm_*.png"
}

main "$@"
