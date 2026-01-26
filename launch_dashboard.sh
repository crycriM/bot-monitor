#!/bin/bash
# Launch script for dashboard (NiceGUI or Streamlit)

# Configuration
CONFIG_FILE="${CONFIG_FILE:-config/web_processor.yml}"
PORT="${PORT:-8880}"
STREAMLIT_PORT="${STREAMLIT_PORT:-8880}"
GW_PORT="${GW_PORT:-14440}"
DASHBOARD="${DASHBOARD:-}"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    echo "Usage: CONFIG_FILE=path/to/config.yml DASHBOARD=nicegui|streamlit $0"
    exit 1
fi

# Show menu if dashboard not specified
if [ -z "$DASHBOARD" ]; then
    echo "Choose which dashboard to launch:"
    echo "1) NiceGUI (default, port $PORT)"
    echo "2) Streamlit (port $STREAMLIT_PORT)"
    echo ""
    read -p "Enter choice (1 or 2): " choice
    case $choice in
        2)
            DASHBOARD="streamlit"
            ;;
        *)
            DASHBOARD="nicegui"
            ;;
    esac
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

cd "$(dirname "$0")"

# Launch selected dashboard
if [ "$DASHBOARD" = "streamlit" ]; then
    echo "Starting Streamlit dashboard..."
    echo "  Config: $CONFIG_FILE"
    echo "  Dashboard Port: $STREAMLIT_PORT"
    echo "  Backend Port: $GW_PORT"
    echo ""
    echo "Open browser to: http://localhost:$STREAMLIT_PORT"
    echo ""
    streamlit run src/web_front_sl.py \
        --server.port "$STREAMLIT_PORT" \
        --logger.level=info \
        -- --config "$CONFIG_FILE" --gw_port "$GW_PORT"
else
    echo "Starting NiceGUI dashboard..."
    echo "  Config: $CONFIG_FILE"
    echo "  Dashboard Port: $PORT"
    echo "  Backend Port: $GW_PORT"
    echo ""
    echo "Open browser to: http://localhost:$PORT"
    echo ""
    python src/web_front_new.py \
        --config "$CONFIG_FILE" \
        --port "$PORT" \
        --gw_port "$GW_PORT"
fi
