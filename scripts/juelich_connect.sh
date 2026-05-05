#!/usr/bin/env bash
# Establish SSH ControlMaster session to Jülich JURECA.
# Run once manually — you will be prompted for TOTP.
# Subsequent connections (including Claude) reuse the socket without TOTP.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Load config: project root first, then HOME fallback
CONFIG_FILE=""
if [[ -f "$REPO_ROOT/.juelich.local" ]]; then
    CONFIG_FILE="$REPO_ROOT/.juelich.local"
elif [[ -f "$HOME/.juelich.local" ]]; then
    CONFIG_FILE="$HOME/.juelich.local"
else
    echo "ERROR: .juelich.local not found."
    echo "       cp $REPO_ROOT/.juelich.local.example $REPO_ROOT/.juelich.local"
    echo "       Then edit it with your username and key path."
    exit 1
fi

# shellcheck source=/dev/null
source "$CONFIG_FILE"

: "${JUELICH_USER:?JUELICH_USER not set in $CONFIG_FILE}"
: "${JUELICH_KEY:?JUELICH_KEY not set in $CONFIG_FILE}"
: "${JUELICH_HOST:?JUELICH_HOST not set in $CONFIG_FILE}"

SOCKET="/tmp/juelich_ctl_${JUELICH_USER}"

# Expand ~ in key path
JUELICH_KEY="${JUELICH_KEY/#\~/$HOME}"

# Check if master connection is already alive
if ssh -o ControlPath="$SOCKET" -O check "${JUELICH_USER}@${JUELICH_HOST}" 2>/dev/null; then
    echo "Already connected to ${JUELICH_HOST} as ${JUELICH_USER} (reusing socket)."
    exit 0
fi

echo "Connecting to ${JUELICH_HOST} as ${JUELICH_USER}..."
echo "You will be prompted for your TOTP code (6 digits)."
echo ""

ssh \
    -i "$JUELICH_KEY" \
    -o ControlMaster=yes \
    -o ControlPath="$SOCKET" \
    -o ControlPersist=4h \
    -o ServerAliveInterval=60 \
    -o ServerAliveCountMax=3 \
    -N -f \
    "${JUELICH_USER}@${JUELICH_HOST}"

# Verify the socket is alive
if ssh -o ControlPath="$SOCKET" -O check "${JUELICH_USER}@${JUELICH_HOST}" 2>/dev/null; then
    echo ""
    echo "Connected to ${JUELICH_HOST} as ${JUELICH_USER}. Socket: ${SOCKET}"
    echo "Connection persists for 4h. Run juelich_exec.sh to send commands."
else
    echo "ERROR: Connection established but socket check failed."
    exit 1
fi
