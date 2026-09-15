#!/usr/bin/env bash
# Chạy TTS server ở chế độ ĐẦY ĐỦ (có VoiceDesign):
# - Load cả 2 model: VoiceDesign (thiết kế giọng bằng instruct) + Base (clone)
# - Dùng khi cần TẠO giọng mới từ mô tả; tốn thêm ~5GB VRAM so với clone-only
# - Tự kill process đang chiếm port trước khi khởi động
#
# Cách dùng:
#   ./scripts/run_full.sh          # port mặc định 8265
#   ./scripts/run_full.sh 8000     # port tùy chọn
set -euo pipefail

PORT="${1:-8265}"
CONDA_ENV="qwen3-tts"

# --- Kill process đang chiếm port (nếu có) ---
PIDS="$(lsof -ti tcp:"$PORT" 2>/dev/null || true)"
if [ -n "$PIDS" ]; then
    echo "Port $PORT đang bị chiếm bởi PID: $PIDS — đang dừng..."
    kill $PIDS 2>/dev/null || true
    # Chờ tối đa 15s cho process thoát êm (server cần thời gian giải phóng VRAM)
    for _ in $(seq 1 15); do
        if [ -z "$(lsof -ti tcp:"$PORT" 2>/dev/null || true)" ]; then
            break
        fi
        sleep 1
    done
    # Vẫn còn sống -> kill cứng
    REMAINING="$(lsof -ti tcp:"$PORT" 2>/dev/null || true)"
    if [ -n "$REMAINING" ]; then
        echo "Process chưa thoát, kill -9: $REMAINING"
        kill -9 $REMAINING 2>/dev/null || true
        sleep 2
    fi
    echo "Port $PORT đã trống."
fi

# --- Kích hoạt conda env ---
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

cd "$(dirname "$0")/.."

echo "Khởi động server chế độ ĐẦY ĐỦ (VoiceDesign + clone) trên port $PORT..."
echo "Kiểm tra khi sẵn sàng:  curl http://127.0.0.1:$PORT/health"
echo "  -> mong đợi: {\"model_loaded\": true, \"clone_model_loaded\": true, ...}"
echo "Khi thiết kế giọng xong, quay về chế độ tiết kiệm VRAM: ./scripts/run_clone_only.sh"
echo

exec uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
