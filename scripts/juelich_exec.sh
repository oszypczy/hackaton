#!/usr/bin/env bash
# Execute a command on Jülich JURECA via existing ControlMaster socket.
# Requires: scripts/juelich_connect.sh to have been run first (TOTP done).
#
# Usage: scripts/juelich_exec.sh "squeue -u $USER"
#        scripts/juelich_exec.sh --force "rm -rf /tmp/old_run"
#
# Safety: destructive patterns are blocked or require confirmation.
# Pass --force as first arg to bypass confirmation prompts (not blocks).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
    FORCE=1
    shift
fi

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 [--force] \"<remote command>\""
    exit 1
fi

CMD="$*"

# Load config
CONFIG_FILE=""
if [[ -f "$REPO_ROOT/.juelich.local" ]]; then
    CONFIG_FILE="$REPO_ROOT/.juelich.local"
elif [[ -f "$HOME/.juelich.local" ]]; then
    CONFIG_FILE="$HOME/.juelich.local"
else
    echo "ERROR: .juelich.local not found. See .juelich.local.example"
    exit 1
fi

# shellcheck source=/dev/null
source "$CONFIG_FILE"

: "${JUELICH_USER:?JUELICH_USER not set}"
: "${JUELICH_KEY:?JUELICH_KEY not set}"
: "${JUELICH_HOST:?JUELICH_HOST not set}"

JUELICH_KEY="${JUELICH_KEY/#\~/$HOME}"
SOCKET="/tmp/juelich_ctl_${JUELICH_USER}"

# Check socket alive
if ! ssh -o ControlPath="$SOCKET" -O check "${JUELICH_USER}@${JUELICH_HOST}" 2>/dev/null; then
    echo "ERROR: No active connection. Run first:"
    echo "       ! scripts/juelich_connect.sh"
    exit 1
fi

# ── Safety checks ────────────────────────────────────────────────────────────

# Patterns that are BLOCKED outright (require --force to override)
BLOCKED_PATTERNS=(
    '\brm\b'
    '\brmdir\b'
    '\bunlink\b'
    'mkfs'
    '\bfdisk\b'
    '\bdd\s+if='
    'git\s+reset\s+--hard'
    'git\s+clean\s+-f'
    '\btruncate\b'
    'scancel\s*$'          # scancel with no job id = cancel ALL
    'chmod\s+[0-7]*7[0-7][0-7]'  # world-writable chmod
    'chown\s'
)

for pattern in "${BLOCKED_PATTERNS[@]}"; do
    if echo "$CMD" | grep -qE "$pattern"; then
        if [[ $FORCE -eq 1 ]]; then
            echo "WARNING: Destructive pattern detected ('$pattern'). --force supplied, proceeding."
        else
            echo "BLOCKED: Command matches destructive pattern: $pattern"
            echo "         If intentional, re-run with --force as first argument."
            echo "         Command: $CMD"
            exit 2
        fi
    fi
done

# Patterns that REQUIRE confirmation (y/N)
CONFIRM_PATTERNS=(
    '\bsbatch\b'
    'scancel\s+[0-9]'
    '\|\s*(bash|sh)\b'
)

if [[ $FORCE -eq 0 ]]; then
    for pattern in "${CONFIRM_PATTERNS[@]}"; do
        if echo "$CMD" | grep -qE "$pattern"; then
            echo "CONFIRM: Command requires approval (matches: $pattern)"
            echo "         Remote command: $CMD"
            printf "         Proceed? [y/N] "
            read -r answer < /dev/tty
            if [[ ! "$answer" =~ ^[Yy]$ ]]; then
                echo "Aborted."
                exit 3
            fi
            break
        fi
    done
fi

# ── Execute ──────────────────────────────────────────────────────────────────
ssh \
    -o ControlMaster=no \
    -o ControlPath="$SOCKET" \
    "${JUELICH_USER}@${JUELICH_HOST}" \
    "$CMD"
