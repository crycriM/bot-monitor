#!/bin/bash
# Launch script for web_front_new.py dashboard

# Configuration
CONFIG_FILE="${CONFIG_FILE:-config/web_processor.yml}"
PORT="${PORT:-8880}"
GW_PORT="${GW_PORT:-14440}"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    echo "Usage: CONFIG_FILE=path/to/config.yml $0"
    exit 1
fi

# Check if backend is running
echo "Checking if backend API is running on port $GW_PORT..."
if ! curl -s "http://localhost:$GW_PORT/status" > /dev/null 2>&1; then
    echo "Warning: Backend API not responding on http://localhost:$GW_PORT"
    echo "Make sure web_api.py is running first!"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Launch dashboard
echo "Starting NiceGUI dashboard..."
echo "  Config: $CONFIG_FILE"
echo "  Dashboard Port: $PORT"
echo "  Backend Port: $GW_PORT"
echo ""
echo "Open browser to: http://localhost:$PORT"
echo ""

cd "$(dirname "$0")"
python src/web_front_new.py \
    --config "$CONFIG_FILE" \
    --port "$PORT" \
    --gw_port "$GW_PORT"
