#!/usr/bin/env bash
# Auto-connect to XProf K8s service via port-forward, then start MCP server.
#
# Env vars (all optional):
#   XPROF_URL          — Override XProf URL (default: auto-detected or http://localhost:$LOCAL_PORT)
#   XPROF_K8S_SERVICE  — K8s service name (default: svc/xprof-service)
#   XPROF_LOCAL_PORT   — Local port for port-forward (default: auto-detect or 8080)
#   XPROF_K8S_PORT     — Remote K8s port (default: 8080)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
K8S_PORT="${XPROF_K8S_PORT:-8080}"
K8S_SVC="${XPROF_K8S_SERVICE:-svc/xprof-service}"

# Health check uses root endpoint (1ms) instead of /runs (90-120s on GCS FUSE).
check_xprof() {
    local url="${1:-$XPROF_URL}"
    curl -s -o /dev/null -w "%{http_code}" -m 3 "$url/" 2>/dev/null | grep -q "200"
}

# Auto-detect existing kubectl port-forward for xprof-service.
detect_existing_forward() {
    local pf_line
    pf_line=$(ps ax -o args= 2>/dev/null | grep -E "kubectl.*port-forward.*xprof" | grep -v grep | head -1)
    if [[ -n "$pf_line" ]]; then
        local port
        port=$(echo "$pf_line" | grep -oE '[0-9]+:[0-9]+' | head -1 | cut -d: -f1)
        if [[ -n "$port" ]]; then
            echo "$port"
            return 0
        fi
    fi
    return 1
}

# --- Resolve XPROF_URL ---
# Priority: XPROF_URL > XPROF_LOCAL_PORT > auto-detect existing port-forward > default 8080

if [[ -n "${XPROF_URL:-}" ]]; then
    echo "Using XPROF_URL=$XPROF_URL" >&2
elif [[ -n "${XPROF_LOCAL_PORT:-}" ]]; then
    export XPROF_URL="http://localhost:$XPROF_LOCAL_PORT"
    echo "Using XPROF_LOCAL_PORT=$XPROF_LOCAL_PORT → $XPROF_URL" >&2
else
    detected_port=$(detect_existing_forward) && {
        export XPROF_URL="http://localhost:$detected_port"
        echo "Auto-detected existing port-forward on port $detected_port → $XPROF_URL" >&2
    } || {
        export XPROF_URL="http://localhost:8080"
    }
fi

LOCAL_PORT="${XPROF_LOCAL_PORT:-$(echo "$XPROF_URL" | grep -oE ':[0-9]+$' | tr -d ':')}"
LOCAL_PORT="${LOCAL_PORT:-8080}"

# 1. Already reachable? Just start server.
if check_xprof "$XPROF_URL"; then
    echo "XProf reachable at $XPROF_URL" >&2
    exec python3 "$SCRIPT_DIR/server.py"
fi

# 2. Port in use but XProf not responding? Wait briefly (may still be starting).
if lsof -i :"$LOCAL_PORT" >/dev/null 2>&1; then
    echo "Port $LOCAL_PORT in use, waiting for XProf..." >&2
    for _ in $(seq 1 10); do
        if check_xprof "$XPROF_URL"; then
            echo "XProf ready at $XPROF_URL" >&2
            exec python3 "$SCRIPT_DIR/server.py"
        fi
        sleep 2
    done
    echo "WARNING: Port $LOCAL_PORT occupied but XProf not responding, starting server anyway" >&2
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
export XPROF_URL="http://localhost:$LOCAL_PORT"

# Clean up port-forward when MCP server exits
trap "kill $PF_PID 2>/dev/null" EXIT

# Wait for XProf to be reachable.
echo "Waiting for XProf to be ready..." >&2
for i in $(seq 1 30); do
    if ! kill -0 "$PF_PID" 2>/dev/null; then
        echo "ERROR: port-forward died. Check kubectl context and service." >&2
        exit 1
    fi
    if check_xprof "$XPROF_URL"; then
        echo "XProf ready at $XPROF_URL" >&2
        exec python3 "$SCRIPT_DIR/server.py"
    fi
    sleep 2
done

echo "ERROR: XProf not ready after 60s. Check service status." >&2
exit 1
