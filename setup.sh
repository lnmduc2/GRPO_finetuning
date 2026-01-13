#!/bin/bash
set -e

echo "--- 🛠️ Khởi tạo môi trường ChatBotSynthetic ---"

# 1. Cài đặt uv nếu chưa có
if ! command -v uv &> /dev/null; then
    echo "📦 Đang cài đặt uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

# 2. Clone và Checkout commit cụ thể
REPO_DIR="ChatBotSynthetic"
if [ ! -d "$REPO_DIR" ]; then
    git clone https://github.com/2Phuong5Nam4/ChatBotSynthetic.git
fi
cd $REPO_DIR
git checkout 38177914ea71bcbbbe0b3edc4ae8fecf799bbfd4


# 3. Kiểm tra GPU để chọn vLLM phù hợp
echo "🔍 Đang check GPU..."
if nvidia-smi | grep -q "Tesla T4"; then
    VLLM_SPEC="vllm==0.9.2 triton==3.2.0"
    echo "✅ Tesla T4 detected: vLLM 0.9.2"
else
    VLLM_SPEC="vllm==0.10.2 triton"
    echo "✅ High-end GPU detected: vLLM 0.10.2"
fi

# 4. Khởi tạo môi trường ảo và cài đặt dependencies
echo "🚀 Đang build venv và sync dependencies..."
uv venv
# Inject vLLM version vào và install mọi thứ
uv add $VLLM_SPEC
uv sync

# 6. Chạy script dataset prepare
echo "📊 Chuẩn bị dataset..."
uv run scripts/dataset_prepare.py

echo "--- ✨ XONG! Chạy 'source .venv/bin/activate' để bắt đầu code. ---"