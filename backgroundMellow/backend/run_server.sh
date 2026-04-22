#!/bin/bash
# Run server script - ensures conda environment is used

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate musicgen

# Change to project directory
cd "$(dirname "$0")"


export PORT="${PORT:-8000}"
export HOST="${HOST:-0.0.0.0}"

echo "Starting Audio Generation API Server..."
echo "Using Python: $(which python)"
echo "Python version: $(python --version)"
echo "Server will be available at http://$HOST:$PORT"
echo "API Documentation: http://$HOST:$PORT/docs"
echo ""
echo "Press CTRL+C to stop the server"
echo ""

# Run uvicorn using the conda environment's Python
python -m uvicorn server:app --host "$HOST" --port "$PORT" --reload

