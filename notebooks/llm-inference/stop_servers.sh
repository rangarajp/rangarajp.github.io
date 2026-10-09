#!/bin/bash

PID_DIR="./pids"

stop_server () {
    NAME=$1
    PID_FILE="$PID_DIR/$2.pid"

    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")

        if kill -0 "$PID" 2>/dev/null; then
            echo "Stopping $NAME (PID $PID)..."
            kill "$PID"
        else
            echo "$NAME is not running."
        fi

        rm -f "$PID_FILE"
    else
        echo "No PID file for $NAME."
    fi
}

stop_server "Gateway" "gateway"
stop_server "Qwen" "qwen"
stop_server "TinyLlama" "tinyllama"

echo "Done."