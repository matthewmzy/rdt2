#!/bin/bash
# RDT2 Docker 管理脚本

set -e

IMAGE_NAME="rdt2-env"
CONTAINER_NAME="rdt2"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_usage() {
    echo -e "${GREEN}RDT2 Docker 管理脚本${NC}"
    echo ""
    echo "用法: ./docker.sh [命令]"
    echo ""
    echo "命令:"
    echo "  build       构建 Docker 镜像"
    echo "  run         运行容器（交互模式）"
    echo "  start       启动容器（后台）"
    echo "  stop        停止容器"
    echo "  exec        进入运行中的容器"
    echo "  logs        查看容器日志"
    echo "  clean       清理容器和镜像"
    echo "  train       启动训练"
    echo "  inference   启动推理服务"
    echo ""
}

build() {
    echo -e "${GREEN}构建 Docker 镜像...${NC}"
    docker build -t ${IMAGE_NAME}:latest .
    echo -e "${GREEN}构建完成！${NC}"
}

run() {
    echo -e "${GREEN}启动容器（交互模式）...${NC}"
    docker run -it --rm \
        --gpus all \
        --name ${CONTAINER_NAME} \
        --shm-size=32g \
        -v $(pwd):/workspace/rdt2 \
        -v ~/.cache/huggingface:/workspace/models/huggingface \
        -v $(pwd)/outputs:/workspace/outputs \
        -e NVIDIA_VISIBLE_DEVICES=all \
        -e HF_HOME=/workspace/models/huggingface \
        -e TRANSFORMERS_CACHE=/workspace/models/huggingface \
        -e WANDB_MODE=offline \
        -w /workspace/rdt2 \
        ${IMAGE_NAME}:latest \
        /bin/bash
}

start() {
    echo -e "${GREEN}后台启动容器...${NC}"
    docker compose up -d rdt2
    echo -e "${GREEN}容器已启动！使用 './docker.sh exec' 进入容器${NC}"
}

stop() {
    echo -e "${YELLOW}停止容器...${NC}"
    docker compose down
    echo -e "${GREEN}容器已停止${NC}"
}

exec_container() {
    echo -e "${GREEN}进入容器...${NC}"
    docker exec -it ${CONTAINER_NAME} /bin/bash
}

logs() {
    docker compose logs -f rdt2
}

clean() {
    echo -e "${YELLOW}清理容器和镜像...${NC}"
    docker compose down --rmi local -v 2>/dev/null || true
    docker rmi ${IMAGE_NAME}:latest 2>/dev/null || true
    echo -e "${GREEN}清理完成${NC}"
}

train() {
    echo -e "${GREEN}启动训练...${NC}"
    docker compose up rdt2-train
}

inference() {
    echo -e "${GREEN}启动推理服务...${NC}"
    docker compose up -d rdt2-inference
    echo -e "${GREEN}推理服务已启动${NC}"
}

# 主逻辑
case "${1}" in
    build)
        build
        ;;
    run)
        run
        ;;
    start)
        start
        ;;
    stop)
        stop
        ;;
    exec)
        exec_container
        ;;
    logs)
        logs
        ;;
    clean)
        clean
        ;;
    train)
        train
        ;;
    inference)
        inference
        ;;
    *)
        print_usage
        ;;
esac
