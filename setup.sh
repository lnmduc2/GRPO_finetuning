#!/bin/bash
set -e

echo "--- 🛠 Đang cấu hình môi trường (vLLM đã có sẵn) ---"

# Cài đặt các gói bổ trợ (KHÔNG cài lại vLLM)
echo "📦 Đang cài đặt Unsloth và các gói phụ trợ..."
uv pip install -qqq --upgrade \
    unsloth triton torchvision bitsandbytes xformers openai pydantic dotenv transformers trl

# Quản lý Repository
REPO_DIR="ChatBotSynthetic"
if [ ! -d "$REPO_DIR" ]; then
    echo "📂 Đang clone repository..."
    git clone https://github.com/2Phuong5Nam4/ChatBotSynthetic.git
fi

# Vào thư mục để checkout và chạy script
cd "$REPO_DIR"
echo "🕒 Đang đồng bộ mã nguồn về commit: 3817791..."
git fetch origin
git checkout 38177914ea71bcbbbe0b3edc4ae8fecf799bbfd4

# Chạy script chuẩn bị dataset
# Vì đã 'cd' vào ChatBotSynthetic nên đường dẫn là scripts/...
echo "📊 Đang chạy dataset_prepare.py..."
if [ -f "scripts/dataset_prepare.py" ]; then
    uv run scripts/dataset_prepare.py
else
    echo "❌ Lỗi: Không tìm thấy file scripts/dataset_prepare.py"
    exit 1
fi

echo "--- ✨ HOÀN THÀNH ---"