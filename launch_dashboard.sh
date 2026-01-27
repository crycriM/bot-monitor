#!/bin/bash
# Launch script for dashboard and/or backend (NiceGUI or Streamlit)

# Python env
source /home/ubuntu/.msenv/bin/activate

# Configuration
EXE_DIR=/home/ubuntu/src/bot-monitor_new
CONFIG_FILE="${CONFIG_FILE:-/home/ubuntu/config/web_processor.yml}"
PORT="${PORT:-8880}"
STREAMLIT_PORT="${STREAMLIT_PORT:-8880}"
GW_PORT="${GW_PORT:-14440}"
DASHBOARD="${DASHBOARD:-}"
LAUNCH_BACKEND="${LAUNCH_BACKEND:-}"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    echo "Usage: CONFIG_FILE=path/to/config.yml DASHBOARD=nicegui|streamlit LAUNCH_BACKEND=yes $0"
    exit 1
fi

# Show menu if both backend and dashboard not specified
if [ -z "$LAUNCH_BACKEND" ] && [ -z "$DASHBOARD" ]; then
    echo "What would you like to launch?"
    echo "1) Backend (web_api)"
    echo "2) NiceGUI Dashboard only"
    echo "3) Streamlit Dashboard only"
    echo "4) Backend + NiceGUI Dashboard"
    echo "5) Backend + Streamlit Dashboard"
    echo ""
    read -p "Enter choice (1-5): " choice
    case $choice in
        1)
            LAUNCH_BACKEND="yes"
            DASHBOARD="none"
            ;;
        2)
            DASHBOARD="nicegui"
            ;;
        3)
            DASHBOARD="streamlit"
            ;;
        4)
            LAUNCH_BACKEND="yes"
            DASHBOARD="nicegui"
            ;;
        5)
            LAUNCH_BACKEND="yes"
            DASHBOARD="streamlit"
            ;;
        *)
            DASHBOARD="nicegui"
            ;;
    esac
fi

# Default to nicegui if only backend specified
if [ "$LAUNCH_BACKEND" = "yes" ] && [ -z "$DASHBOARD" ]; then
    DASHBOARD="nicegui"
fi

# Launch backend if requested
if [ "$LAUNCH_BACKEND" = "yes" ]; then
    echo "Starting backend API (web_api.py)..."
    echo "  Config: $CONFIG_FILE"
    echo "  Backend Port: $GW_PORT"
    echo ""
    
    python $EXE_DIR/src/web_api.py --config "$CONFIG_FILE" &
    BACKEND_PID=$!
    echo "Backend started with PID: $BACKEND_PID"
    
    # Give backend time to start
    sleep 2
    echo ""
fi

# Skip dashboard launch if only backend was requested
if [ "$DASHBOARD" = "none" ]; then
    echo "Backend running. Press Ctrl+C to stop."
    wait $BACKEND_PID
    exit 0
fi

# Check if backend is running (if we're launching dashboard)
#if [ "$DASHBOARD" != "none" ]; then
#    echo "Checking if backend API is running on port $GW_PORT..."
#    if ! curl -s "http://localhost:$GW_PORT/status" > /dev/null 2>&1; then
#        echo "Warning: Backend API not responding on http://localhost:$GW_PORT"
#        echo "Make sure web_api.py is running first!"
#        echo ""
#        read -p "Continue anyway? (y/N) " -n 1 -r
#        echo
#        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
#            if [ ! -z "$BACKEND_PID" ]; then
#                kill $BACKEND_PID 2>/dev/null
#            fi
#            exit 1
#        fi
#    fi
#fi

# Launch selected dashboard
if [ "$DASHBOARD" = "streamlit" ]; then
    echo "Starting Streamlit dashboard..."
    echo "  Config: $CONFIG_FILE"
    echo "  Dashboard Port: $STREAMLIT_PORT"
    echo "  Backend Port: $GW_PORT"
    echo ""
    echo "Open browser to: http://localhost:$STREAMLIT_PORT"
    echo ""
    
    streamlit run $EXE_DIR/src/web_front_sl.py \
        --server.port "$STREAMLIT_PORT" \
        --logger.level=info \
        -- --config "$CONFIG_FILE" --gw_port "$GW_PORT"
    
    # Kill backend if we started it
    if [ ! -z "$BACKEND_PID" ]; then
        echo "Stopping backend..."
        kill $BACKEND_PID 2>/dev/null
    fi
# elif [ "$DASHBOARD" = "nicegui" ]; then
#     echo "Starting NiceGUI dashboard..."
#     echo "  Config: $CONFIG_FILE"
#     echo "  Dashboard Port: $PORT"
#     echo "  Backend Port: $GW_PORT"
#     echo ""
#     echo "Open browser to: http://localhost:$PORT"
#     echo ""
    
#     python $EXE_DIR/src/web_front_new.py \
#         --config "$CONFIG_FILE" \
#         --port "$PORT" \
#         --gw_port "$GW_PORT"
    
#     # Kill backend if we started it
#     if [ ! -z "$BACKEND_PID" ]; then
#         echo "Stopping backend..."
#         kill $BACKEND_PID 2>/dev/null
#     fi
fi

# LAUNCH_BACKEND=yes DASHBOARD=streamlit ./launch_dashboard.sh
# LAUNCH_BACKEND=yes DASHBOARD=nicegui ./launch_dashboard.sh
# LAUNCH_BACKEND=yes DASHBOARD=none ./launch_dashboard.sh  # Backend only