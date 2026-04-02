#!/usr/bin/env bash
# Auto-connect to XProf K8s service via port-forward, then start MCP server.
#
# Env vars (all optional):
#   XPROF_URL          — Override XProf URL (default: http://localhost:$LOCAL_PORT)
#   XPROF_K8S_SERVICE  — K8s service name (default: svc/xprof-service)
#   XPROF_LOCAL_PORT   — Local port for port-forward (default: 8080)
#   XPROF_K8S_PORT     — Remote K8s port (default: 8080)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_PORT="${XPROF_LOCAL_PORT:-8080}"
K8S_PORT="${XPROF_K8S_PORT:-8080}"
K8S_SVC="${XPROF_K8S_SERVICE:-svc/xprof-service}"
export XPROF_URL="${XPROF_URL:-http://localhost:$LOCAL_PORT}"

check_xprof() {
    curl -s --compressed -m 5 "$XPROF_URL/runs" >/dev/null 2>&1
}

# 1. Already reachable? Just start server.
if check_xprof; then
    exec python3 "$SCRIPT_DIR/server.py"
fi

# 2. Port in use but XProf not responding? Wait briefly.
if lsof -i :"$LOCAL_PORT" >/dev/null 2>&1; then
    echo "Port $LOCAL_PORT in use, waiting for XProf..." >&2
    for _ in $(seq 1 15); do
        if check_xprof; then
            exec python3 "$SCRIPT_DIR/server.py"
        fi
        sleep 2
    done
    echo "WARNING: Port $LOCAL_PORT occupied but XProf not responding, starting anyway" >&2
    exec python3 "$SCRIPT_DIR/server.py"
fi

# 3. Start port-forward automatically.
if ! command -v kubectl >/dev/null 2>&1; then
    echo "WARNING: kubectl not found, cannot auto port-forward. Start XProf manually." >&2
    exec python3 "$SCRIPT_DIR/server.py"
fi

echo "Starting kubectl port-forward $K8S_SVC $LOCAL_PORT:$K8S_PORT ..." >&2
kubectl port-forward "$K8S_SVC" "$LOCAL_PORT:$K8S_PORT" >/dev/null 2>&1 &
PF_PID=$!

# Clean up port-forward when MCP server exits
trap "kill $PF_PID 2>/dev/null" EXIT

# Wait for XProf to be reachable (initial scan takes ~80s on GCS FUSE)
echo "Waiting for XProf to be ready..." >&2
for i in $(seq 1 60); do
    if ! kill -0 "$PF_PID" 2>/dev/null; then
        echo "ERROR: port-forward died. Check kubectl context and service." >&2
        exit 1
    fi
    if check_xprof; then
        echo "XProf ready on $XPROF_URL" >&2
        exec python3 "$SCRIPT_DIR/server.py"
    fi
    sleep 2
done

echo "ERROR: XProf not ready after 120s. Check service status." >&2
exit 1
