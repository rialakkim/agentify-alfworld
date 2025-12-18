# Agentify ALFWorld

**A standardized, multi-dimensional agent assessment framework for the ALFWorld benchmark**

This project transforms the [ALFWorld](https://github.com/alfworld/alfworld) benchmark into a standardized agent evaluation system using the A2A (Agent-to-Agent) protocol. Unlike the original benchmark that only evaluates binary task completion, this implementation provides comprehensive behavioral quality metrics including cleanup behavior, repetition efficiency, and navigation cycles.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Overview](#overview)
- [Architecture](#architecture)
- [Evaluation Metrics](#evaluation-metrics)
- [Task Types](#task-types)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Validation & Testing](#validation--testing)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [References](#references)

---

## Quick Start

### One-Command Evaluation

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# Run a complete evaluation
python main.py launch
```

This single command will:
1. Start the green agent (assessment manager) on port 9001
2. Start the white agent (target being tested) on port 9002
3. Run one ALFWorld game
4. Display comprehensive evaluation results
5. Terminate both agents

### More Examples

```bash
# Run 5 games
python main.py launch --num-games 5

# Test specific task types
python main.py launch --task-types "1,2,3" --num-games 3

# Comprehensive evaluation (5 games per task type = 30 total)
python main.py launch --coverage-mode comprehensive --num-games-per-type 5
```

---

## Installation

### Step 1: Prerequisites

- **Python 3.9 or higher**
- **OpenAI API Key** (or compatible LLM provider)

### Step 2: Install Dependencies

```bash
# Clone or navigate to the project
cd agentify-alfworld

# Create and activate virtual environment
python3.9 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .
```

### Step 3: Setup ALFWorld

```bash
# Patch TextWorld (required for ALFWorld compatibility)
python src/scripts/patch_textworld.py

# Download ALFWorld dataset (this may take a while)
alfworld-download
```

### Step 4: Configure API Key

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# Optional: Configure white agent model
export WHITE_AGENT_MODEL="gpt-4o-mini"  # or "gpt-4o"
export WHITE_AGENT_TEMPERATURE="0.0"
```

### Step 5: Verify Installation

```bash
# Test that everything works
python main.py launch --num-games 1
```

If you see evaluation results, installation is complete!

---

## Overview

### What is This?

This project **agentifies** the ALFWorld benchmark, meaning it:
- Uses the A2A protocol for standardized agent communication
- Evaluates agents on household tasks in interactive TextWorld environments
- Provides multi-dimensional metrics beyond simple success/failure
- Works with any A2A-compatible agent (no ALFWorld-specific code needed)

### Key Features

✅ **Multi-Dimensional Evaluation**: Tracks task completion AND behavioral quality  
✅ **Process-Oriented Metrics**: Evaluates HOW agents complete tasks, not just IF they complete them  
✅ **Standardized Protocol**: Uses A2A protocol for cross-platform compatibility  
✅ **Fine-Grained Scoring**: 0-100 score with detailed breakdown  
✅ **Per-Task-Type Analysis**: Performance breakdown across all 6 task types  
✅ **Comprehensive Coverage**: Flexible test case distribution strategies

### Improvements Over Original ALFWorld

The original ALFWorld benchmark only evaluates binary task completion (pass/fail). This implementation adds:

1. **Behavioral Quality Metrics**: Cleanup behavior, repetition efficiency, cycle detection
2. **Process Evaluation**: Tracks action sequences and intermediate behavior
3. **Standardized Protocol**: A2A protocol enables cross-platform evaluation
4. **Detailed Analysis**: Per-task-type breakdowns and comprehensive logging

---

## Architecture

The system consists of three main components:

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

### Components Explained

1. **Green Agent** (Assessment Manager)
   - Manages the ALFWorld environment internally
   - Sends observations and available actions to white agent via A2A protocol
   - Executes actions chosen by white agent
   - Tracks all actions and computes behavioral metrics
   - Generates comprehensive evaluation reports

2. **White Agent** (Target Being Tested)
   - Receives task descriptions and environment observations
   - Uses LLM to reason about tasks and choose actions
   - Has NO knowledge of ALFWorld internals
   - Returns action commands as plain text
   - Can be any A2A-compatible agent

3. **Launcher Script**
   - Orchestrates the complete evaluation workflow
   - Starts both green and white agents
   - Manages the evaluation process
   - Displays results and cleans up

---

## Evaluation Metrics

The green agent provides a comprehensive multi-dimensional evaluation system:

### Task Performance Metrics

- **Success Rate**: Percentage of games successfully completed
- **Average Score**: Mean completion score across all games (0.0 to 1.0)
- **Average Steps**: Mean number of steps per game
- **Time Used**: Total wall-clock time for assessment

### Behavioral Quality Metrics

#### 1. Cleanup Behavior (20% of overall score)

Measures whether agents properly close containers they open.

- **Metric**: Cleanup Rate = (Items Correctly Closed) / (Total Items Opened)
- **Example**: Agent opens fridge, cools object, closes fridge → 100% cleanup rate
- **Penalty**: Leaving containers open (e.g., fridge, drawers) reduces score

#### 2. Repetition Efficiency (20% of overall score)

Measures how often agents repeat consecutive identical actions.

- **Metric**: Repetition Rate = (Repeated Actions) / (Total Actions)
- **Example**: Agent repeats "take apple" 3 times → high repetition rate
- **Penalty**: Excessive repetition indicates poor action selection

#### 3. Cycle Efficiency (20% of overall score)

Detects repetitive navigation patterns (e.g., going back and forth between locations).

- **Metric**: Cycle Rate = (Actions in Cycles) / (Total Actions)
- **Example**: Agent cycles between "go to couch" and "go to table" repeatedly
- **Penalty**: Navigation cycles indicate poor exploration planning

### Overall Score Calculation

The overall white agent score (0-100) is calculated as:

```
Overall Score = 
  Completion Score (40%) +
  Cleanup Score (20%) +
  Repetition Score (20%) +
  Cycle Score (20%)
```

Where:
- **Completion Score**: (Success Rate × 100) × 0.4
- **Cleanup Score**: (Cleanup Rate × 100) × 0.2
- **Repetition Score**: ((1 - Repetition Rate) × 100) × 0.2
- **Cycle Score**: ((1 - Cycle Rate) × 100) × 0.2

### Example Score Breakdown

**Perfect Agent** (100/100):
- Task completed: 40 points
- Perfect cleanup: 20 points
- No repetitions: 20 points
- No cycles: 20 points

**Agent with Cleanup Issue** (80/100):
- Task completed: 40 points
- Forgot to close fridge: 0 points
- No repetitions: 20 points
- No cycles: 20 points

**Agent with Repetition** (95/100):
- Task completed: 40 points
- Perfect cleanup: 20 points
- Some repetition: 15 points
- No cycles: 20 points

---

## Task Types

ALFWorld includes 6 different task types:

| ID | Task Type | Description | Example |
|----|-----------|-------------|---------|
| 1 | `pick_and_place_simple` | Pick up an object and place it in a receptacle | "put apple in garbagecan" |
| 2 | `look_at_obj_in_light` | Examine an object under a light source | "look at book under desklamp" |
| 3 | `pick_clean_then_place_in_recep` | Clean an object and place it somewhere | "put a clean cup in drawer" |
| 4 | `pick_heat_then_place_in_recep` | Heat an object and place it somewhere | "put a hot potato in fridge" |
| 5 | `pick_cool_then_place_in_recep` | Cool an object and place it somewhere | "put a cool apple in garbagecan" |
| 6 | `pick_two_obj_and_place` | Pick up two objects and place them | "put apple and book in drawer" |

---

## Usage Examples

### Basic Usage

```bash
# Run 1 game (default)
python main.py launch

# Run 5 games
python main.py launch --num-games 5

# Run with custom max steps
python main.py launch --num-games 3 --max-steps 30
```

### Task Type Selection

```bash
# Test only simple pick and place tasks
python main.py launch --task-types "1" --num-games 5

# Test transformation tasks (clean, heat, cool)
python main.py launch --task-types "3,4,5" --num-games 9

# Test all task types (default)
python main.py launch --task-types "1,2,3,4,5,6" --num-games 6
```

### Coverage Modes

```bash
# Standard mode: Auto-balanced distribution
python main.py launch --num-games 12  # 2 games per task type

# Balanced mode: Explicit per-task-type count
python main.py launch --num-games-per-type 3  # 3 games per type = 18 total

# Comprehensive mode: Deep testing
python main.py launch --coverage-mode comprehensive --num-games-per-type 5  # 30 total
```

### Data Splits

```bash
# Use out-of-distribution split (default, most challenging)
python main.py launch --train-eval eval_out_of_distribution

# Use in-distribution split
python main.py launch --train-eval eval_in_distribution

# Use training split (for development)
python main.py launch --train-eval train
```

### Running Agents Separately

For development or testing custom white agents:

```bash
# Terminal 1: Start green agent
python main.py green

# Terminal 2: Start white agent
export OPENAI_API_KEY="your-key"
python main.py white

# Terminal 3: Send custom assessment requests using A2A client
```

### White Agent Configuration

```bash
# Use different model
export WHITE_AGENT_MODEL="gpt-4o"
python main.py launch

# Adjust temperature
export WHITE_AGENT_TEMPERATURE="0.5"
python main.py launch

# Disable enhanced reasoning
export WHITE_AGENT_REASONING="false"
python main.py launch
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (required) | Your OpenAI API key |
| `WHITE_AGENT_MODEL` | `gpt-4o-mini` | Model to use (e.g., `gpt-4o`, `gpt-4o-mini`) |
| `WHITE_AGENT_TEMPERATURE` | `0.0` | Temperature for LLM responses |
| `WHITE_AGENT_REASONING` | `true` | Enable enhanced reasoning |
| `LITELLM_PROXY_API_KEY` | (optional) | If using LiteLLM proxy |

### Command Line Options

```bash
python main.py launch [OPTIONS]

Options:
  --env-type TEXT              Environment type (default: AlfredTWEnv)
  --train-eval TEXT            Data split: train, eval_in_distribution, eval_out_of_distribution
  --num-games INTEGER          Number of games to evaluate (default: 1)
  --max-steps INTEGER          Maximum steps per game (default: 50)
  --task-types TEXT            Comma-separated task type IDs (1-6), e.g., "1,2,3"
  --coverage-mode TEXT         Coverage mode: standard, balanced, comprehensive
  --num-games-per-type INTEGER Number of games per task type
```

---

## Validation & Testing

### Reproducing Original Benchmark Results

To verify faithful reproduction of the original ALFWorld benchmark:

```bash
# Run 100 games matching original benchmark scale
python main.py launch \
  --train-eval eval_out_of_distribution \
  --num-games 100 \
  --max-steps 50 \
  --task-types "1,2,3,4,5,6"
```

Expected results (baseline white agent with gpt-4o-mini):
- Success Rate: ~25% (matches original baseline)
- Average Steps: ~28 steps per game
- Results saved to `logs/run_YYYYMMDD_HHMMSS/run_summary.json`

### Validation Test Cases

Three validation test cases are provided to verify evaluation accuracy:

**Test Case 1: Simple Pick and Place**
```bash
python main.py launch --num-games 1 --task-types "1" --max-steps 50
```
Validates basic task completion and perfect behavioral metrics.

**Test Case 2: Cool and Place with Cleanup**
```bash
python main.py launch --num-games 1 --task-types "5" --max-steps 50
```
Validates cleanup metric detection when agents forget to close containers.

**Test Case 3: Repetition Detection**
```bash
python main.py launch --num-games 1 --task-types "1" --max-steps 50
```
Validates repetition metric detection when agents repeat actions.

**Run All Test Cases:**
```bash
python main.py launch --num-games 3 --task-types "1,5,1" --max-steps 50
```

### Viewing Results

All evaluation results are saved to timestamped log directories:

```
logs/
└── run_20240101_120000/
    ├── run_summary.json      # Aggregate metrics
    ├── run_report.txt        # Human-readable report
    ├── game_000.json         # Individual game results
    ├── game_001.json
    └── ...
```

---

## Project Structure

```
agentify-alfworld/
├── main.py                      # CLI entry point
├── pyproject.toml               # Project configuration
├── configs/
│   └── base_config.yaml         # ALFWorld configuration
├── src/
│   ├── __init__.py
│   ├── launcher.py              # Assessment launcher
│   ├── green_agent/
│   │   ├── __init__.py
│   │   ├── agent.py             # Green agent implementation
│   │   └── alfworld_green_agent.toml
│   ├── white_agent/
│   │   ├── __init__.py
│   │   └── agent.py             # White agent implementation
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── behavior_metrics.py  # Behavioral metrics computation
│   │   └── run_logger.py        # Logging utilities
│   └── my_util/
│       ├── __init__.py
│       └── my_a2a.py            # A2A client utilities
└── README.md
```

---

## How It Works

### 1. Agent Standardization (A2A Protocol)

The white agent only needs to support the A2A message protocol. It receives:
- Environment observations (text)
- Available actions list
- Task instructions

And returns:
- Selected action command (text)

### 2. Self-Explanatory Tasks

Each task is designed to be understandable without ALFWorld-specific knowledge:
- Clear goal descriptions (e.g., "put a cool apple in garbagecan")
- Numbered list of available actions
- Simple response format requirements

### 3. Benchmark Agentification

The green agent:
- Manages the ALFWorld environment internally
- Translates environment state to text for the white agent
- Executes actions and tracks behavior
- Evaluates performance and reports comprehensive metrics

### Evaluation Process

1. Green agent resets ALFWorld environment and gets initial observation
2. Green agent sends task description and observation to white agent via A2A
3. White agent receives message, reasons about task, chooses action
4. White agent returns action command to green agent
5. Green agent executes action in ALFWorld environment
6. Steps 2-5 repeat until task completes or max steps reached
7. Green agent computes behavioral metrics from action sequence
8. Green agent aggregates metrics across all games and generates report

---

## Custom White Agents

To test your own agent, implement an A2A-compatible agent:

```python
from a2a.server.agent_execution import AgentExecutor
from a2a.utils import new_agent_text_message

class MyCustomAgent(AgentExecutor):
    async def execute(self, context, event_queue):
        user_input = context.get_user_input()
        
        # Your reasoning logic here
        action = decide_action(user_input)
        
        # Return action to green agent
        await event_queue.enqueue_event(
            new_agent_text_message(action, context_id=context.context_id)
        )
```

### Message Format

The green agent sends messages in this format:

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

The white agent should respond with just the action:
```
go to desk 1
```

---

## Troubleshooting

### Common Issues

**Issue: "No games found"**
```bash
# Solution: Download ALFWorld dataset
alfworld-download
```

**Issue: "API Key not found"**
```bash
# Solution: Set OPENAI_API_KEY
export OPENAI_API_KEY="your-key-here"
```

**Issue: "Port already in use"**
```bash
# Solution: Use different ports
python main.py green --port 9003
python main.py white --port 9004
```

**Issue: Import errors**
```bash
# Solution: Reinstall package
pip install -e .
```

---

## References

- [ALFWorld Paper](https://arxiv.org/abs/2010.03768) - Original ALFWorld benchmark
- [ALFWorld GitHub](https://github.com/alfworld/alfworld) - Official ALFWorld repository
- [A2A Protocol](https://github.com/google/a2a) - Agent-to-Agent protocol specification
- [Agentify the Agent Assessment Blog](https://agentbeats.github.io/) - Agentification approach
