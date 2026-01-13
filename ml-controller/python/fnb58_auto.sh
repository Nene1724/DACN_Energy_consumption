#!/bin/bash
# FNB58 Auto Measurement - Bash Wrapper
# Chạy fnb58_auto.py với giao diện CLI đơn giản

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/fnb58_auto.py"

# Kiểm tra fnb58_auto.py tồn tại
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "❌ Lỗi: Không tìm thấy $PYTHON_SCRIPT"
    exit 1
fi

# Kiểm tra fnb58_reader.py tồn tại
if [ ! -f "$SCRIPT_DIR/fnb58_reader.py" ]; then
    echo "❌ Lỗi: Không tìm thấy fnb58_reader.py"
    echo "Hãy copy fnb58_reader.py vào $SCRIPT_DIR"
    exit 1
fi

# Kiểm tra Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Lỗi: Python3 không tìm thấy"
    exit 1
fi

# In help
show_help() {
    echo "=== FNB58 Auto Measurement ==="
    echo ""
    echo "Cách dùng:"
    echo "  ./fnb58_auto.sh                    # Chạy với config mặc định (30s)"
    echo "  ./fnb58_auto.sh -d 60              # Đo 60 giây"
    echo "  ./fnb58_auto.sh -p /dev/ttyUSB0    # Chỉ định cổng"
    echo "  ./fnb58_auto.sh -i 192.168.1.50    # Chỉ định IP agent"
    echo "  ./fnb58_auto.sh -l                 # Chỉ đo cục bộ (không trigger agent)"
    echo ""
    echo "Tùy chọn:"
    echo "  -d DURATION    Thời gian đo (giây, mặc định 30)"
    echo "  -p PORT        Cổng serial (ví dụ /dev/ttyUSB0)"
    echo "  -i IP          IP/hostname agent (mặc định localhost)"
    echo "  --port PORT    Cổng agent (mặc định 8000)"
    echo "  -s SERVER      URL server (mặc định http://localhost:5000)"
    echo "  -m MODEL       Tên model"
    echo "  -t DEVICE      Loại thiết bị (jetson_nano, rpi5, bbb)"
    echo "  -l             Chỉ đo FNB58 cục bộ"
    echo "  -h             Xem trợ giúp"
    echo ""
}

# Parse arguments
DURATION=30
PORT=""
AGENT_IP="localhost"
AGENT_PORT=8000
SERVER="http://localhost:5000"
MODEL_NAME=""
DEVICE_TYPE="jetson_nano"
LOCAL_MEASURE=0

while getopts "d:p:i:s:m:t:lh" opt; do
    case $opt in
        d) DURATION="$OPTARG" ;;
        p) PORT="--port $OPTARG" ;;
        i) AGENT_IP="$OPTARG" ;;
        s) SERVER="$OPTARG" ;;
        m) MODEL_NAME="--model-name $OPTARG" ;;
        t) DEVICE_TYPE="--device-type $OPTARG" ;;
        l) LOCAL_MEASURE="--local-measure" ;;
        h) show_help; exit 0 ;;
        *) show_help; exit 1 ;;
    esac
done

# Chạy Python script
echo "🚀 Bắt đầu FNB58 Auto Measurement..."
echo ""

python3 "$PYTHON_SCRIPT" \
    --duration "$DURATION" \
    --agent-ip "$AGENT_IP" \
    --agent-port "$AGENT_PORT" \
    --server "$SERVER" \
    $PORT \
    $MODEL_NAME \
    $DEVICE_TYPE \
    $LOCAL_MEASURE

exit $?
