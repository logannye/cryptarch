#!/bin/bash
# Wrapper: activates venv + runs the engine. Invoked by LaunchAgent.
#
# Logs are written by LaunchAgent's StandardOutPath/StandardErrorPath
# (data/cryptarch_stdout.log and data/cryptarch_stderr.log).

set -euo pipefail

cd "$HOME/cryptarch"

# Wait for PostgreSQL to be available (handle cold-boot race after macOS reboot).
PSQL=/opt/homebrew/opt/postgresql@16/bin/psql
WAIT_BUDGET=60
ELAPSED=0
until $PSQL -d cryptarch -c "SELECT 1" >/dev/null 2>&1; do
    if [ "$ELAPSED" -ge "$WAIT_BUDGET" ]; then
        echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) Postgres not reachable after ${WAIT_BUDGET}s, exiting" >&2
        exit 1
    fi
    sleep 2
    ELAPSED=$((ELAPSED + 2))
done

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) Starting cryptarch..."
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) Cryptarch running as PID $$"

# Prevent the Mac from sleeping while the bot is running.
exec /usr/bin/caffeinate -i -s -w $$ \
    "$HOME/cryptarch/.venv/bin/python3" -m cryptarch run
