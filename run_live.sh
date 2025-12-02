#!/bin/bash
# Live trading worker startup script
# Usage: ./run_live.sh [start|stop|restart|status|logs]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/logs/worker.log"
PID_FILE="$SCRIPT_DIR/logs/worker.pid"

start() {
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "Worker already running (PID: $(cat "$PID_FILE"))"
        return 1
    fi

    mkdir -p "$SCRIPT_DIR/logs"

    echo "Starting live worker..."
    cd "$SCRIPT_DIR" && nohup bash -c 'PYTHONPATH=src python3 -m runner.main' > "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"

    sleep 2
    if kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "Worker started (PID: $(cat "$PID_FILE"))"
        echo "Logs: tail -f $LOG_FILE"
    else
        echo "Failed to start worker. Check logs: $LOG_FILE"
        return 1
    fi
}

stop() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo "Stopping worker (PID: $PID)..."
            kill "$PID"
            sleep 2
            if kill -0 "$PID" 2>/dev/null; then
                echo "Force killing..."
                kill -9 "$PID"
            fi
            echo "Worker stopped"
        else
            echo "Worker not running (stale PID file)"
        fi
        rm -f "$PID_FILE"
    else
        # Fallback: kill by name
        pkill -f "runner.main" && echo "Worker stopped" || echo "Worker not running"
    fi
}

status() {
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "Worker running (PID: $(cat "$PID_FILE"))"
    else
        echo "Worker not running"
    fi
}

logs() {
    tail -f "$LOG_FILE"
}

case "${1:-start}" in
    start)   start ;;
    stop)    stop ;;
    restart) stop; sleep 1; start ;;
    status)  status ;;
    logs)    logs ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs}"
        exit 1
        ;;
esac
