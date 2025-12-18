#!/bin/bash
# Script to start the white agent locally

echo "=========================================="
echo "Starting White Agent Locally"
echo "=========================================="
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  Warning: .env file not found"
    echo "   Creating .env from .env.example..."
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "   Please edit .env and add your OPENAI_API_KEY"
        echo ""
    else
        echo "   Please create a .env file with your OPENAI_API_KEY"
        echo "   Example:"
        echo "   echo 'OPENAI_API_KEY=your-key-here' > .env"
        echo ""
    fi
fi

# Check if OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ]; then
    # Try to load from .env
    if [ -f .env ]; then
        export $(cat .env | grep -v '^#' | xargs)
    fi
fi

if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ Error: OPENAI_API_KEY not set"
    echo "   Please set it in your .env file or export it:"
    echo "   export OPENAI_API_KEY='your-key-here'"
    exit 1
fi

echo "✓ API Key configured"
echo ""

# Display configuration
MODEL=${WHITE_AGENT_MODEL:-gpt-4o-mini}
TEMPERATURE=${WHITE_AGENT_TEMPERATURE:-0.0}
REASONING=${WHITE_AGENT_REASONING:-true}
HOST=${HOST:-localhost}
PORT=${PORT:-9002}

echo "Configuration:"
echo "  Model: $MODEL"
echo "  Temperature: $TEMPERATURE"
echo "  Enhanced Reasoning: $REASONING"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo ""
echo "Starting white agent on http://$HOST:$PORT"
echo "Press Ctrl+C to stop"
echo ""
echo "=========================================="
echo ""

# Start the white agent
python main.py white --host "$HOST" --port "$PORT"


