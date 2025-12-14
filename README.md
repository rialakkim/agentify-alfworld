# Agentify ALFWorld

This project demonstrates how to **agentify** the [ALFWorld](https://github.com/alfworld/alfworld) benchmark using the A2A (Agent-to-Agent) protocol for standardized agent assessment.

## Overview

ALFWorld is a benchmark for evaluating agents on interactive TextWorld environments that parallel embodied worlds in the ALFRED dataset. This agentified version allows any A2A-compatible agent to be tested on ALFWorld tasks without requiring ALFWorld-specific code.

### Architecture

The system consists of three components:

1. **Green Agent** (Assessment Manager): Manages the ALFWorld environment, sends observations to the white agent, and evaluates performance.

2. **White Agent** (Target Being Tested): A general-purpose LLM agent that receives task descriptions and chooses actions. It has NO knowledge of ALFWorld internals.

3. **Launcher Script**: Orchestrates the entire assessment workflow with a single command.

```
┌─────────────────┐         ┌─────────────────┐
│   Green Agent   │◄───────►│   White Agent   │
│  (Assessment)   │  A2A    │    (Target)     │
│                 │ Protocol│                 │
│  - ALFWorld Env │         │  - LLM-based    │
│  - Metrics      │         │  - No ALFWorld  │
│  - Evaluation   │         │    knowledge    │
└─────────────────┘         └─────────────────┘
        │
        ▼
   Assessment
    Results
```

## Prerequisites

1. **Python 3.9+**

2. **ALFWorld Data**: Download the ALFWorld game files:
   ```bash
   pip install alfworld[full]
   alfworld-download
   ```

3. **OpenAI API Key** (or compatible LLM provider):
   ```bash
   export OPENAI_API_KEY="your-api-key"
   ```

## Installation

```bash
# Clone or navigate to the project
cd agentify-alfworld

# Install dependencies
pip install -e .

# Or with uv
uv pip install -e .
```

## Quick Start

### One-Command Evaluation

Run the complete assessment workflow:

```bash
python main.py launch
```

This will:
1. Start the green agent on port 9001
2. Start the white agent on port 9002
3. Run one ALFWorld game
4. Display assessment results
5. Terminate both agents

### Configuration Options

```bash
# Run 5 games with max 30 steps each
python main.py launch --num-games 5 --max-steps 30

# Only test specific task types
python main.py launch --task-types "1,2,3"

# Use a different data split
python main.py launch --train-eval eval_in_distribution
```

### Running Agents Separately

For development or testing custom white agents:

```bash
# Terminal 1: Start green agent
python main.py green

# Terminal 2: Start white agent
python main.py white

# Terminal 3: Send assessment request via your own client
```

## Task Types

ALFWorld includes 6 task types:

| ID | Task Type | Description |
|----|-----------|-------------|
| 1 | pick_and_place_simple | Pick up an object and place it in a receptacle |
| 2 | look_at_obj_in_light | Examine an object under a light source |
| 3 | pick_clean_then_place_in_recep | Clean an object and place it somewhere |
| 4 | pick_heat_then_place_in_recep | Heat an object and place it somewhere |
| 5 | pick_cool_then_place_in_recep | Cool an object and place it somewhere |
| 6 | pick_two_obj_and_place | Pick up two objects and place them |

## White Agent Configuration

The white agent uses LiteLLM and can be configured via environment variables:

```bash
export WHITE_AGENT_MODEL="openai/gpt-4o"      # Model to use
export WHITE_AGENT_PROVIDER="openai"           # Provider
export WHITE_AGENT_TEMPERATURE="0.0"           # Temperature
```

## Custom White Agents

To test your own agent, implement an A2A-compatible agent that:

1. Accepts the A2A `SendMessage` protocol
2. Receives text observations and available actions
3. Returns an action command as text

Example message from green agent:
```
ENVIRONMENT OBSERVATION:
You are in the middle of a room. Looking around you see...

Available actions (choose one):
  1. go to desk 1
  2. go to shelf 1
  3. go to drawer 1
  ...

Please provide your next action:
```

Expected white agent response:
```
go to desk 1
```

## Metrics

The assessment reports:

- **Success Rate**: Percentage of games won
- **Average Score**: Mean score across games
- **Average Steps**: Mean steps per game
- **Total Time**: Wall-clock time for assessment
- **Per-game Results**: Detailed results for each game

## Project Structure

```
agentify-alfworld/
├── main.py                 # CLI entry point
├── pyproject.toml          # Project configuration
├── configs/
│   └── base_config.yaml    # ALFWorld configuration
├── src/
│   ├── __init__.py
│   ├── launcher.py         # Assessment launcher
│   ├── green_agent/
│   │   ├── __init__.py
│   │   ├── agent.py        # Green agent implementation
│   │   └── alfworld_green_agent.toml
│   ├── white_agent/
│   │   ├── __init__.py
│   │   └── agent.py        # White agent implementation
│   └── my_util/
│       ├── __init__.py     # Tag parsing utilities
│       └── my_a2a.py       # A2A client utilities
└── README.md
```

## How It Works

Following the approach from "Agentify the Agent Assessment":

### 1. Agent Standardization (A2A Protocol)

The white agent only needs to support the A2A message protocol. It receives:
- Environment observations (text)
- Available actions list
- Task instructions

And returns:
- Selected action command (text)

### 2. Self-Explanatory Tasks

Each task is designed to be understandable without ALFWorld-specific knowledge:
- Clear goal descriptions
- Numbered list of available actions
- Simple response format requirements

### 3. Benchmark Agentification

The green agent:
- Manages the ALFWorld environment internally
- Translates environment state to text for the white agent
- Evaluates performance and reports metrics

## Extending

### Adding New White Agents

Create a new agent that implements the A2A protocol:

```python
from a2a.server.agent_execution import AgentExecutor

class MyCustomAgent(AgentExecutor):
    async def execute(self, context, event_queue):
        user_input = context.get_user_input()
        # Your reasoning logic here
        action = decide_action(user_input)
        await event_queue.enqueue_event(
            new_agent_text_message(action, context_id=context.context_id)
        )
```

### Modifying Assessment Logic

Edit `src/green_agent/agent.py` to:
- Change observation formatting
- Add new metrics
- Modify evaluation criteria

## References

- [ALFWorld Paper](https://arxiv.org/abs/2010.03768)
- [ALFWorld GitHub](https://github.com/alfworld/alfworld)
- [A2A Protocol](https://github.com/google/a2a)
- [Agentify the Agent Assessment Blog](https://agentbeats.github.io/)

## License

MIT License
