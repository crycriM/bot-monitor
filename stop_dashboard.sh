#!/bin/bash
# Stop script for backend and dashboard using PID files

# Function to kill process from PID file
kill_from_pid_file() {
    local pid_file=$1
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if [ -n "$pid" ] && ps -p $pid > /dev/null 2>&1; then
            echo "Killing process from $pid_file (PID: $pid)..."
            kill $pid 2>/dev/null
            sleep 2
            if ps -p $pid > /dev/null 2>&1; then
                echo "Process still running, using SIGKILL..."
                kill -9 $pid 2>/dev/null
            fi
        else
            echo "No valid process found in $pid_file."
        fi
        rm -f "$pid_file"
    else
        echo "PID file $pid_file not found."
    fi
}

# Kill backend
kill_from_pid_file /tmp/backend.pid

# Kill dashboard
kill_from_pid_file /tmp/dashboard.pid

echo "Stop script completed."