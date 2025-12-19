"""Launcher module - initiates and coordinates the ALFWorld evaluation process.

This script:
1. Starts the green agent (assessment manager)
2. Starts the white agent (target being tested)
3. Sends the assessment task to the green agent
4. Collects and displays results
5. Terminates both agents
"""

import multiprocessing
import json
from src.green_agent.agent import start_green_agent
from src.white_agent.agent import start_white_agent
from src.white_agent.repetitive_agent import start_repetitive_white_agent
from src.white_agent.alternating_agent import start_alternating_white_agent
from src.my_util import my_a2a


async def launch_evaluation(
    env_type: str = "AlfredTWEnv",
    train_eval: str = "eval_out_of_distribution",
    num_games: int = 1,
    max_steps: int = 1,
    task_types: list = None,
    coverage_mode: str = "standard",
    num_games_per_type: int = None,
    white_agent_type: str = "standard",
):
    """
    Launch the complete ALFWorld evaluation workflow.
    
    Args:
        env_type: Environment type ('AlfredTWEnv', 'AlfredThorEnv', or 'AlfredHybrid')
        train_eval: Data split to use ('train', 'eval_in_distribution', 'eval_out_of_distribution')
        num_games: Number of games to evaluate
        max_steps: Maximum steps per game
        task_types: List of task type IDs to evaluate (1-6)
        coverage_mode: Coverage mode ('standard', 'balanced', 'comprehensive', 'adversarial')
        num_games_per_type: Number of games per task type (overrides num_games if set)
        white_agent_type: Type of white agent to test ('standard' or 'repetitive')
    """
    if task_types is None:
        task_types = [1, 2, 3, 4, 5, 6]
    
    print("=" * 60)
    print("ALFWorld Agentified Assessment")
    print("=" * 60)
    
    print("\nLaunching green agent (assessment manager)...")
    green_address = ("localhost", 9001)
    green_url = f"http://{green_address[0]}:{green_address[1]}"
    p_green = multiprocessing.Process(
        target=start_green_agent, args=("alfworld_green_agent", *green_address)
    )
    p_green.start()
    assert await my_a2a.wait_agent_ready(green_url, timeout=30), "Green agent not ready in time"
    print("✓ Green agent is ready.")

    print("\nLaunching white agent (target being tested)...")
    white_address = ("localhost", 9002)
    white_url = f"http://{white_address[0]}:{white_address[1]}"
    
    if white_agent_type == "repetitive":
        print("  Using: Repetitive White Agent (always picks first action)")
        p_white = multiprocessing.Process(
            target=start_repetitive_white_agent, args=("repetitive_white_agent", *white_address)
        )
    elif white_agent_type == "alternating":
        print("  Using: Alternating White Agent (alternates between first and second actions)")
        p_white = multiprocessing.Process(
            target=start_alternating_white_agent, args=("alternating_white_agent", *white_address)
        )
    else:
        print("  Using: Standard White Agent")
        p_white = multiprocessing.Process(
            target=start_white_agent, args=("alfworld_white_agent", *white_address)
        )
    
    p_white.start()
    assert await my_a2a.wait_agent_ready(white_url, timeout=30), "White agent not ready in time"
    print("✓ White agent is ready.")

    print("\n" + "-" * 60)
    print("Sending assessment task to green agent...")
    print("-" * 60)
    
    task_config = {
        "env_type": env_type,
        "train_eval": train_eval,
        "num_games": num_games,
        "max_steps": max_steps,
        "task_types": task_types,
        "coverage_mode": coverage_mode,
        "num_games_per_type": num_games_per_type,
    }
    
    task_text = f"""
Your task is to run ALFWorld assessment to test the agent located at:
<white_agent_url>
http://{white_address[0]}:{white_address[1]}/
</white_agent_url>
You should use the following env configuration:
<env_config>
{json.dumps(task_config, indent=2)}
</env_config>
    """
    
    print("\nTask description:")
    print(task_text)
    print("\nSending task to green agent...")
    
    response = await my_a2a.send_message(green_url, task_text)
    
    print("\n" + "=" * 60)
    print("ASSESSMENT RESULTS")
    print("=" * 60)
    
    # Extract the actual text from the response
    from a2a.types import SendMessageSuccessResponse, Message
    from a2a.utils import get_text_parts
    
    if hasattr(response, 'root') and isinstance(response.root, SendMessageSuccessResponse):
        result = response.root.result
        if isinstance(result, Message):
            text_parts = get_text_parts(result.parts)
            for text in text_parts:
                print(text)
        else:
            print(response)
    else:
        print(response)

    print("\n" + "-" * 60)
    print("Evaluation complete. Terminating agents...")
    p_green.terminate()
    p_green.join()
    p_white.terminate()
    p_white.join()
    print("✓ Agents terminated successfully.")
    print("=" * 60)
