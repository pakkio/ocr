#!/bin/bash

# Kill any process running on port 7860
echo "🔍 Checking for processes on port 7860..."

# Find processes using port 7860
PIDS=$(lsof -ti:7860 2>/dev/null)

if [ -z "$PIDS" ]; then
    echo "✅ No processes found running on port 7860"
else
    echo "💀 Found processes on port 7860: $PIDS"
    
    # Kill the processes
    for PID in $PIDS; do
        echo "🗡️  Killing process $PID..."
        kill -9 $PID
    done
    
    # Wait a moment and check again
    sleep 2
    REMAINING=$(lsof -ti:7860 2>/dev/null)
    
    if [ -z "$REMAINING" ]; then
        echo "✅ Port 7860 is now free"
    else
        echo "⚠️  Some processes may still be running: $REMAINING"
    fi
fi

# Also kill any gradio_app.py processes
echo "🔍 Checking for gradio_app.py processes..."
GRADIO_PIDS=$(pgrep -f "gradio_app.py" 2>/dev/null)

if [ -z "$GRADIO_PIDS" ]; then
    echo "✅ No gradio_app.py processes found"
else
    echo "💀 Found gradio_app.py processes: $GRADIO_PIDS"
    pkill -f "gradio_app.py"
    echo "✅ Killed gradio_app.py processes"
fi

echo "🎯 Port 7860 cleanup complete!"