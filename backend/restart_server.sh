#!/bin/bash
# Auto-restart script for stream_server.py
# This script is called when modules are installed to restart the server

echo "Restarting stream server..."

# Find and kill existing stream_server.py process
pkill -f "python.*stream_server.py" 2>/dev/null

# Wait a moment for cleanup
sleep 2

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Start the server in the background with virtual environment
cd "$SCRIPT_DIR"
source venv/bin/activate
python stream_server.py >> /tmp/backend.log 2>&1 &

echo "Stream server restarted (PID: $!)"
echo "Logs: /tmp/backend.log"
