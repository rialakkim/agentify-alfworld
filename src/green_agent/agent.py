"""Green agent implementation - manages ALFWorld assessment and evaluation."""

import uvicorn
import tomllib
import os
import json
import time
import yaml
from dotenv import load_dotenv
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, SendMessageSuccessResponse, Message
from a2a.utils import new_agent_text_message, get_text_parts
from src.my_util import parse_tags, my_a2a
from src.metrics import BehaviorMetrics, ActionTracker, RunLogger

import pathlib
env_path = pathlib.Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path, override=True)

RESPOND_ACTION_NAME = "respond"

TASK_TYPES = {
    1: "pick_and_place_simple",
    2: "look_at_obj_in_light",
    3: "pick_clean_then_place_in_recep",
    4: "pick_heat_then_place_in_recep",
    5: "pick_cool_then_place_in_recep",
    6: "pick_two_obj_and_place"
}


def load_agent_card_toml(agent_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(f"{current_dir}/{agent_name}.toml", "rb") as f:
        return tomllib.load(f)


def load_alfworld_config(config_path=None):
    """Load ALFWorld configuration from YAML file."""
    if config_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "..", "..", "configs", "base_config.yaml")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_alfworld_env(config, train_eval="eval_out_of_distribution"):
    """Create an ALFWorld TextWorld environment."""
    from alfworld.agents.environment import get_environment
    
    env_type = config['env']['type']
    AlfredEnv = get_environment(env_type)
    alfred_env = AlfredEnv(config, train_eval=train_eval)
    env = alfred_env.init_env(batch_size=1)
    return env, alfred_env


def format_admissible_commands(commands):
    """Format admissible commands as a numbered list for the agent."""
    if not commands or len(commands) == 0:
        return "No admissible commands available."
    
    cmd_list = commands[0] if isinstance(commands[0], list) else commands
    formatted = ["Available actions (choose one):"]
    for i, cmd in enumerate(cmd_list, 1):
        formatted.append(f"  {i}. {cmd}")
    return "\n".join(formatted)


async def ask_agent_to_solve(white_agent_url, env, alfred_env, max_steps=50, game_idx=0, logger: RunLogger = None):
    """
    Run a single ALFWorld episode with the white agent.
    
    The green agent:
    1. Resets the environment and gets initial observation
    2. Sends task description and observation to white agent
    3. Receives action from white agent
    4. Executes action in environment
    5. Repeats until done or max steps reached
    6. Computes behavior metrics for the white agent
    """
    total_steps = 0
    obs, info = env.reset()
    
    # Initialize action tracker for behavior metrics
    action_tracker = ActionTracker()
    
    initial_obs = obs[0] if isinstance(obs, (list, tuple)) else obs
    admissible_commands = info.get('admissible_commands', [[]])
    
    task_description = f"""You are an agent in a household environment. Your goal is to complete tasks by taking actions.

ENVIRONMENT OBSERVATION:
{initial_obs}

{format_admissible_commands(admissible_commands)}

INSTRUCTIONS:
- Read the observation carefully to understand your current situation and goal.
- Choose an action from the available actions list.
- Respond with ONLY the exact action text (e.g., "go to desk 1" or "take book 1 from desk 1").
- Do NOT include any explanation, just the action command.

Please provide your first action:"""

    next_green_message = task_description
    context_id = None
    done = False
    won = False
    score = 0
    
    for step in range(max_steps):
        total_steps = step + 1
        
        print(f"@@@ Green agent: Step {step + 1}, sending to white agent...")
        print(f"Message preview: {next_green_message[:200]}...")
        
        try:
            time.sleep(2)  # Delay to avoid rate limiting
            white_agent_response = await my_a2a.send_message(
                white_agent_url, next_green_message, context_id=context_id
            )
            res_root = white_agent_response.root
            assert isinstance(res_root, SendMessageSuccessResponse)
            res_result = res_root.result
            assert isinstance(res_result, Message)
            
            if context_id is None:
                context_id = res_result.context_id
            
            text_parts = get_text_parts(res_result.parts)
            assert len(text_parts) >= 1, "Expecting at least one text part from the white agent"
            white_text = text_parts[0].strip()
            print(f"@@@ White agent response: {white_text}")
            
            action = white_text.strip().lower()
            action_clean = action.replace('"', '').replace("'", "").strip()
            
            if action_clean.startswith("action:"):
                action_clean = action_clean[7:].strip()
            
            admissible_list = admissible_commands[0] if admissible_commands else []
            admissible_lower = [cmd.lower() for cmd in admissible_list]
            
            if action_clean not in admissible_lower:
                best_match = None
                for cmd in admissible_list:
                    if action_clean in cmd.lower() or cmd.lower() in action_clean:
                        best_match = cmd
                        break
                
                if best_match is None and admissible_list:
                    for i, char in enumerate(action_clean):
                        if char.isdigit():
                            prefix = action_clean[:i].strip()
                            for cmd in admissible_list:
                                if cmd.lower().startswith(prefix):
                                    best_match = cmd
                                    break
                            break
                
                if best_match:
                    action_clean = best_match.lower()
                else:
                    print(f"Warning: Action '{action_clean}' not in admissible commands, using first available")
                    if admissible_list:
                        action_clean = admissible_list[0].lower()
            
            for cmd in admissible_list:
                if cmd.lower() == action_clean:
                    action_to_execute = cmd
                    break
            else:
                action_to_execute = action_clean
            
            print(f"@@@ Executing action: {action_to_execute}")
            obs, scores, dones, infos = env.step([action_to_execute])
            
            # Track the action for behavior metrics
            current_obs = obs[0] if isinstance(obs, (list, tuple)) else obs
            action_tracker.add_action(action_to_execute, current_obs)
            
            # Log the action if logger is available
            if logger:
                admissible_list = infos.get('admissible_commands', [[]])
                logger.log_action(
                    game_idx=game_idx,
                    step=step + 1,
                    action=action_to_execute,
                    observation=current_obs,
                    admissible_commands=admissible_list[0] if admissible_list else []
                )
            
            done = dones[0] if isinstance(dones, (list, tuple)) else dones
            score = scores[0] if isinstance(scores, (list, tuple)) else scores
            won = infos.get('won', [False])[0] if isinstance(infos.get('won', [False]), list) else infos.get('won', False)
            admissible_commands = infos.get('admissible_commands', [[]])
            
            if done:
                print(f"@@@ Episode finished! Won: {won}, Score: {score}")
                break
            
            next_green_message = f"""OBSERVATION after your action "{action_to_execute}":
{current_obs}

{format_admissible_commands(admissible_commands)}

Please provide your next action (just the action command, no explanation):"""
            
        except Exception as e:
            print(f"@@@ Error in step {step + 1}: {e}")
            import traceback
            traceback.print_exc()
            break
    
    # Compute behavior metrics
    behavior_metrics = BehaviorMetrics.compute_all_metrics(
        tracker=action_tracker,
        task_completed=won,
        max_steps=max_steps,
    )
    
    return {
        "won": won,
        "score": score,
        "steps": total_steps,
        "game_idx": game_idx,
        "behavior_metrics": behavior_metrics,
    }


class AlfWorldGreenAgentExecutor(AgentExecutor):
    def __init__(self):
        pass

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        print("Green agent: Received a task, parsing...")
        user_input = context.get_user_input()
        tags = parse_tags(user_input)
        
        # Support both tagged format (from launcher) and direct format (from AgentBeats)
        white_agent_url = tags.get("white_agent_url", "http://localhost:9002/")
        env_config_str = tags.get("env_config", None)
        
        if not env_config_str:
            try:
                env_config = json.loads(user_input)
                print(f"Parsed config directly from input: {env_config}")
            except:
                print("Warning: Could not parse env config. Using defaults.")
                env_config = {
                    "env_type": "AlfredTWEnv",
                    "train_eval": "eval_out_of_distribution",
                    "num_games": 1,
                    "max_steps": 50,
                    "task_types": [1, 2, 3, 4, 5, 6],
                }
        else:
            env_config = json.loads(env_config_str)

        print("Green agent: Setting up the ALFWorld environment...")
        
        train_eval = env_config.get("train_eval", "eval_out_of_distribution")
        max_steps = env_config.get("max_steps", 50)
        num_games = env_config.get("num_games", 1)
        
        # COVERAGE EXPANSION: Multiple configuration modes
        coverage_mode = env_config.get("coverage_mode", "standard")
        num_games_per_type = env_config.get("num_games_per_type", None)
        
        if coverage_mode == "comprehensive":
            num_games_per_type = env_config.get("num_games_per_type", 5)
            num_games = num_games_per_type * 6
            print(f"COMPREHENSIVE COVERAGE: {num_games_per_type} games × 6 task types = {num_games} total test cases")
        elif num_games_per_type:
            num_games = num_games_per_type * 6
            print(f"BALANCED COVERAGE: {num_games_per_type} games per task type ({num_games} total)")
        else:
            num_games_per_type = max(1, num_games // 6) if num_games >= 6 else 1
            print(f"STANDARD MODE: ~{num_games_per_type} games per task type")
        
        # Load ALFWorld config and apply settings AFTER coverage mode adjusts num_games
        config = load_alfworld_config()
        
        if "task_types" in env_config:
            config['env']['task_types'] = env_config['task_types']
        config['dataset']['num_eval_games'] = num_games  # Use adjusted num_games
        
        env, alfred_env = create_alfworld_env(config, train_eval=train_eval)
        
        # Initialize run logger
        logger = RunLogger()
        logger.set_run_metadata({
            "env_config": env_config,
            "train_eval": train_eval,
            "max_steps": max_steps,
            "num_games": num_games,
            "white_agent_url": white_agent_url,
        })
        
        metrics = {
            "total_games": num_games,
            "wins": 0,
            "total_score": 0,
            "total_steps": 0,
            "game_results": [],
            "coverage_mode": coverage_mode,
            "task_type_breakdown": {
                f"task_{i}": {
                    "name": TASK_TYPES[i].replace("_", " ").title(),
                    "wins": 0,
                    "total": 0,
                    "scores": [],
                    "steps": []
                }
                for i in range(1, 7)
            }
        }

        if alfred_env.num_games == 0:
            await event_queue.enqueue_event(
                new_agent_text_message(
                    "ERROR: No games found! Please run 'alfworld-download' to download the game data."
                )
            )
            return

        actual_num_games = min(num_games, alfred_env.num_games)
        if actual_num_games < num_games:
            print(f"Warning: Only {alfred_env.num_games} games available, running {actual_num_games} instead of {num_games}")

        print("Green agent: Starting evaluation...")
        timestamp_started = time.time()

        # Coverage Expansion: Track count per task type
        task_type_counts = {i: 0 for i in range(1, 7)}

        for game_idx in range(actual_num_games):
            # Determine which task type this game tests
            task_type_id = ((game_idx // max(1, num_games_per_type)) % 6) + 1
            task_type_counts[task_type_id] += 1
            
            print(f"\n@@@ Starting game {game_idx + 1}/{actual_num_games} (Task Type {task_type_id}: {TASK_TYPES[task_type_id]})")
            
            result = await ask_agent_to_solve(
                white_agent_url, env, alfred_env, 
                max_steps=max_steps, game_idx=game_idx,
                logger=logger
            )
            
            # Extract per-game rates from behavior metrics
            bm = result.get("behavior_metrics", {})
            cleanup_rate = bm.get("cleanup", {}).get("cleanup_rate", 1.0)
            repetition_rate = bm.get("repeated_steps", {}).get("repetition_rate", 0.0)
            cycle_rate = bm.get("cycles", {}).get("cycle_rate", 0.0)
            
            result["cleanup_rate"] = cleanup_rate
            result["repetition_rate"] = repetition_rate
            result["cycle_rate"] = cycle_rate
            
            metrics["game_results"].append(result)
            if result["won"]:
                metrics["wins"] += 1
            metrics["total_score"] += result["score"]
            metrics["total_steps"] += result["steps"]
            
            # Track per-task-type statistics for coverage analysis
            task_key = f"task_{task_type_id}"
            metrics["task_type_breakdown"][task_key]["total"] += 1
            if result["won"]:
                metrics["task_type_breakdown"][task_key]["wins"] += 1
            metrics["task_type_breakdown"][task_key]["scores"].append(result["score"])
            metrics["task_type_breakdown"][task_key]["steps"].append(result["steps"])
            
            # Log game result with behavior metrics
            logger.log_game(game_idx, result)
            
            # Print behavior metrics summary for this game
            bm = result.get("behavior_metrics", {})
            summary_bm = bm.get("summary", {})
            print(f"@@@ Game {game_idx + 1} result: won={result['won']}, score={result['score']}, steps={result['steps']}")
            print(f"    Behavior Score: {summary_bm.get('overall_score', 0):.1f}/100 (Grade: {summary_bm.get('grade', 'N/A')})")

        metrics["time_used"] = time.time() - timestamp_started
        metrics["success_rate"] = metrics["wins"] / max(1, metrics["total_games"])
        metrics["avg_score"] = metrics["total_score"] / max(1, metrics["total_games"])
        metrics["avg_steps"] = metrics["total_steps"] / max(1, metrics["total_games"])
        
        # Calculate average rates across all games
        cleanup_rates = [r.get("cleanup_rate", 1.0) for r in metrics["game_results"]]
        repetition_rates = [r.get("repetition_rate", 0.0) for r in metrics["game_results"]]
        cycle_rates = [r.get("cycle_rate", 0.0) for r in metrics["game_results"]]
        
        metrics["avg_cleanup_rate"] = sum(cleanup_rates) / len(cleanup_rates) if cleanup_rates else 1.0
        metrics["avg_repetition_rate"] = sum(repetition_rates) / len(repetition_rates) if repetition_rates else 0.0
        metrics["avg_cycle_rate"] = sum(cycle_rates) / len(cycle_rates) if cycle_rates else 0.0
        
        # Calculate overall white agent score (0-100) based on wins and behavior metrics
        completion_weight = 40.0
        cleanup_weight = 20.0
        repetition_weight = 20.0
        cycle_weight = 20.0
        
        completion_score = (metrics["success_rate"] * 100) * (completion_weight / 100)
        cleanup_score = (metrics["avg_cleanup_rate"] * 100) * (cleanup_weight / 100)
        repetition_score = (1.0 - metrics["avg_repetition_rate"]) * 100 * (repetition_weight / 100)
        cycle_score = (1.0 - metrics["avg_cycle_rate"]) * 100 * (cycle_weight / 100)
        
        overall_white_agent_score = completion_score + cleanup_score + repetition_score + cycle_score
        
        metrics["overall_white_agent_score"] = round(overall_white_agent_score, 2)
        metrics["score_breakdown"] = {
            "completion_score": round(completion_score, 2),
            "cleanup_score": round(cleanup_score, 2),
            "repetition_score": round(repetition_score, 2),
            "cycle_score": round(cycle_score, 2),
        }
        
        # Aggregate behavior metrics across all games
        metrics["behavior_metrics"] = RunLogger.aggregate_behavior_metrics(metrics["game_results"])

        env.close()
        
        # Finalize logging
        log_path = logger.finalize_run(metrics)

        success = metrics["wins"] > 0
        result_emoji = "Completed Task" if success else "Failed to complete task"

        print("Green agent: Evaluation complete.")
        
        # Build per-task-type breakdown
        task_breakdown = metrics.get("task_type_breakdown", {})
        coverage_mode_display = metrics.get("coverage_mode", "standard")
        
        task_type_details = "COVERAGE ANALYSIS\n"
        task_type_details += f"   Coverage Mode: {coverage_mode_display.upper()}\n"
        task_type_details += "══════════════════════════════════════════════════════\n"
        
        for task_key in [f"task_{i}" for i in range(1, 7)]:
            stats = task_breakdown.get(task_key, {})
            task_name = stats.get("name", "Unknown")
            total = stats.get("total", 0)
            wins = stats.get("wins", 0)
            success_rate = (wins / total * 100) if total > 0 else 0
            avg_score = sum(stats.get("scores", [0])) / max(1, len(stats.get("scores", [0])))
            avg_steps = sum(stats.get("steps", [0])) / max(1, len(stats.get("steps", [0])))
            
            task_type_details += f"\n{task_name} ({task_key})\n"
            task_type_details += f"  Test Cases: {total} games\n"
            task_type_details += f"  Success Rate: {wins}/{total} ({success_rate:.1f}%)\n"
            task_type_details += f"  Avg (Completion)Score: {avg_score:.2f}\n"
            task_type_details += f"  Avg Steps: {avg_steps:.1f}\n"
        
        summary = f"""Finished ALFWorld Assessment.

Results Summary: {result_emoji}
══════════════════════════════════════════

TASK PERFORMANCE
- Total Games: {metrics['total_games']}
- Wins: {metrics['wins']}/{metrics['total_games']}
- Success Rate: {metrics['success_rate']:.2%}
- Average Score: {metrics['avg_score']:.2f}
- Average Steps: {metrics['avg_steps']:.1f}
- Time Used: {metrics['time_used']:.2f}s

{task_type_details}

BEHAVIOR METRICS (AVERAGED ACROSS ALL GAMES)
══════════════════════════════════════════

Repeated Steps:
   - Average repetition rate: {metrics['avg_repetition_rate']:.1%}

Cleanup Behavior:
   - Average cleanup rate: {metrics['avg_cleanup_rate']:.1%}

Action Cycles:
   - Average cycle rate: {metrics['avg_cycle_rate']:.1%}

OVERALL WHITE AGENT SCORE
══════════════════════════════════════════
   Score: {metrics['overall_white_agent_score']:.1f}/100

   Score Breakdown:
   - Completion (Task Success): {metrics['score_breakdown']['completion_score']:.1f}/40.0
   - Cleanup Quality: {metrics['score_breakdown']['cleanup_score']:.1f}/20.0
   - Repetition Efficiency: {metrics['score_breakdown']['repetition_score']:.1f}/20.0
   - Cycle Efficiency: {metrics['score_breakdown']['cycle_score']:.1f}/20.0

Detailed logs saved to: {log_path}
"""
        
        await event_queue.enqueue_event(
            new_agent_text_message(summary)
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError


def start_green_agent(agent_name="alfworld_green_agent", host="localhost", port=9001):
    print("Starting ALFWorld green agent...")
    agent_card_dict = load_agent_card_toml(agent_name)

    # Use AGENT_URL if set (for online platform), otherwise construct from host/port
    agent_url = os.getenv("AGENT_URL")
    if agent_url is None:
        agent_url = f"http://{host}:{port}"
    agent_card_dict["url"] = agent_url

    request_handler = DefaultRequestHandler(
        agent_executor=AlfWorldGreenAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=AgentCard(**agent_card_dict),
        http_handler=request_handler,
    )

    uvicorn.run(app.build(), host=host, port=port)
