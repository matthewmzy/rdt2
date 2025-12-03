#!/bin/bash
# ============================================================
# RDT2 Flow Matching 完整 Episode 离线推理脚本
# ============================================================
# 
# 该脚本实现对完整 episode 的滚动预测：
#   - 每次预测 24 帧的相对动作
#   - 取前 step_size (默认 12) 帧累积到轨迹
#   - 下一次预测从该帧继续，使用 GT 观测
#   - 最终得到完整 episode 的预测轨迹并与 GT 对比
#
# 用法:
#   1. 默认参数推理 episode0:
#      bash scripts/inference_pika_fm_full.sh
#   
#   2. 指定 episode 和 step_size:
#      bash scripts/inference_pika_fm_full.sh --episode episode1 --step-size 6
#   
#   3. 指定 checkpoint:
#      bash scripts/inference_pika_fm_full.sh --checkpoint 10000 --episode episode0
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
INFERENCE_OUTPUT_DIR="inference_outputs_fm_full"

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

run_full_episode_inference() {
    local checkpoint_path="$1"
    local episode="$2"
    local step_size="$3"
    
    print_header "Running Full Episode Inference"
    echo "FM Checkpoint: $checkpoint_path"
    echo "VLM Model: $VLM_MODEL"
    echo "Episode: $episode"
    echo "Step Size: $step_size"
    echo ""
    
    python pika_test_scripts/inference_offline_fm_full.py \
        --fm-checkpoint "$checkpoint_path" \
        --vlm-model "$VLM_MODEL" \
        --config-path "$CONFIG_PATH" \
        --normalizer-path "$NORMALIZER_PATH" \
        --shard-dir "$SHARD_DIR" \
        --episode "$episode" \
        --step-size "$step_size" \
        --output-dir "$INFERENCE_OUTPUT_DIR"
}

# ===================== 使用说明 =====================

show_usage() {
    echo "用法: bash scripts/inference_pika_fm_full.sh [选项]"
    echo ""
    echo "选项:"
    echo "  --checkpoint <step>   指定 checkpoint step (默认: 最新)"
    echo "  --episode <name>      指定 episode 名称 (默认: episode0)"
    echo "  --step-size <n>       每次执行的步数 (默认: 12)"
    echo "  --help                显示此帮助信息"
    echo ""
    echo "说明:"
    echo "  该脚本实现对完整 episode 的滚动预测:"
    echo "  - 每次预测 24 帧的相对动作"
    echo "  - 取前 step_size 帧累积到轨迹"
    echo "  - 使用 GT 观测继续下一次预测"
    echo "  - 最终对比预测轨迹和 GT 轨迹"
    echo ""
    echo "示例:"
    echo "  bash scripts/inference_pika_fm_full.sh                           # 默认参数"
    echo "  bash scripts/inference_pika_fm_full.sh --episode episode1        # 指定 episode"
    echo "  bash scripts/inference_pika_fm_full.sh --step-size 6             # 减小步长"
    echo "  bash scripts/inference_pika_fm_full.sh --checkpoint 10000        # 指定 checkpoint"
}

# ===================== 主流程 =====================

main() {
    cd "$(dirname "$0")/.."  # 切换到项目根目录
    
    # 默认参数
    CHECKPOINT_STEP=""
    EPISODE="episode0"
    STEP_SIZE=12
    
    # 解析参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            --checkpoint)
                CHECKPOINT_STEP="$2"
                shift 2
                ;;
            --episode)
                EPISODE="$2"
                shift 2
                ;;
            --step-size)
                STEP_SIZE="$2"
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
    
    # 运行推理
    run_full_episode_inference "$CHECKPOINT_PATH" "$EPISODE" "$STEP_SIZE"
    
    print_header "Done!"
    echo "输出目录: $INFERENCE_OUTPUT_DIR"
    echo ""
    echo "查看结果:"
    echo "  - 可视化图片: ${INFERENCE_OUTPUT_DIR}/full_episode_*.png"
    echo "  - 轨迹数据:   ${INFERENCE_OUTPUT_DIR}/full_episode_*.npz"
}

main "$@"
