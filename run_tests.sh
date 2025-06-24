#!/bin/bash
# 🧪 Automated Test Runner for Gradio OCR Interface
# =================================================
# 
# This script starts the Gradio server and runs automated client tests
# to validate the centralized model family system and battle interfaces.

set -e  # Exit on any error

echo "🚀 Starting Gradio OCR Test Suite"
echo "================================="

# Check if poetry is available
if ! command -v poetry &> /dev/null; then
    echo "❌ Poetry not found. Please install poetry first."
    exit 1
fi

# Check environment variables
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "⚠️ WARNING: OPENROUTER_API_KEY not set. Some tests may fail."
fi

# Start Gradio server in background
echo "🔄 Starting Gradio server..."
poetry run python gradio_main.py &
GRADIO_PID=$!

# Wait for server to start
echo "⏳ Waiting for server to initialize..."
sleep 10

# Check if server is running
if ! kill -0 $GRADIO_PID 2>/dev/null; then
    echo "❌ Failed to start Gradio server"
    exit 1
fi

echo "✅ Gradio server started (PID: $GRADIO_PID)"

# Run client tests
echo "🧪 Running automated client tests..."
poetry run python test_client.py
TEST_EXIT_CODE=$?

# Clean up
echo "🧹 Cleaning up..."
kill $GRADIO_PID 2>/dev/null || true
wait $GRADIO_PID 2>/dev/null || true

# Report results
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "🎉 All tests passed successfully!"
else
    echo "❌ Some tests failed. Check the output above."
fi

exit $TEST_EXIT_CODE