# White Agent Local Setup Guide

This guide will help you run your white agent locally.

## Quick Start

### Option 1: Using the Helper Script (Recommended)

```bash
# Make sure you have your API key set
export OPENAI_API_KEY="your-api-key-here"

# Or create a .env file with:
# OPENAI_API_KEY=your-api-key-here

# Run the helper script
./start_white_agent.sh
```

### Option 2: Using Python CLI Directly

```bash
# Set your API key
export OPENAI_API_KEY="your-api-key-here"

# Start the white agent
python main.py white
```

### Option 3: Using Environment Variables

```bash
# Set all configuration via environment variables
export OPENAI_API_KEY="your-api-key-here"
export WHITE_AGENT_MODEL="gpt-4o-mini"
export WHITE_AGENT_TEMPERATURE="0.0"
export WHITE_AGENT_REASONING="true"

# Start the agent
python main.py white --host localhost --port 9002
```

## Configuration Options

The white agent can be configured using environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (required) | Your OpenAI API key |
| `WHITE_AGENT_MODEL` | `gpt-4o-mini` | Model to use (e.g., `gpt-4o`, `gpt-4o-mini`) |
| `WHITE_AGENT_TEMPERATURE` | `0.0` | Temperature for LLM responses |
| `WHITE_AGENT_REASONING` | `true` | Enable enhanced reasoning |
| `LITELLM_PROXY_API_KEY` | (optional) | If using LiteLLM proxy instead |

## Testing Your White Agent

### Test 1: Standalone (Agent Only)

Just start the white agent:
```bash
python main.py white
```

The agent will be available at `http://localhost:9002`

### Test 2: With Green Agent (Full Assessment)

In **Terminal 1** - Start the green agent:
```bash
python main.py green
```

In **Terminal 2** - Start the white agent:
```bash
python main.py white
```

In **Terminal 3** - Run a full assessment:
```bash
python main.py launch --num-games 1 --max-steps 50
```

### Test 3: Custom Configuration

```bash
# Use a different model
export WHITE_AGENT_MODEL="gpt-4o"
python main.py white

# Use a different port
python main.py white --port 9003

# Use a different host
python main.py white --host 0.0.0.0 --port 9002
```

## Troubleshooting

### Issue: "API Key not found"
**Solution**: Make sure you've set `OPENAI_API_KEY`:
```bash
export OPENAI_API_KEY="your-key-here"
# Or create a .env file in the project root
```

### Issue: "Port already in use"
**Solution**: Use a different port:
```bash
python main.py white --port 9003
```

### Issue: "Module not found"
**Solution**: Install dependencies:
```bash
pip install -e .
```

## What the White Agent Does

The white agent:
1. Receives task descriptions and observations from the green agent
2. Uses an LLM to reason about the task and choose actions
3. Tracks state (location, holding items, task progress)
4. Returns action commands to be executed in ALFWorld

The agent has **NO knowledge** of ALFWorld internals - it only follows the task instructions provided, making it a truly self-explanatory assessment.

## Next Steps

Once your white agent is running:
- Test it with the green agent using `python main.py launch`
- Customize the agent logic in `src/white_agent/agent.py`
- Adjust the system prompt or reasoning logic for better performance
- Monitor the agent's behavior and improve based on results


