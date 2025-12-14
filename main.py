"""CLI entry point for agentify-alfworld."""

import typer
import asyncio
from typing import List, Optional

from src.green_agent import start_green_agent
from src.white_agent import start_white_agent
from src.launcher import launch_evaluation

app = typer.Typer(
    help="Agentified ALFWorld - Standardized agent assessment framework for ALFWorld benchmark"
)


@app.command()
def green(
    host: str = typer.Option("localhost", help="Host address for the green agent"),
    port: int = typer.Option(9001, help="Port for the green agent"),
):
    """Start the green agent (assessment manager)."""
    start_green_agent(host=host, port=port)


@app.command()
def white(
    host: str = typer.Option("localhost", help="Host address for the white agent"),
    port: int = typer.Option(9002, help="Port for the white agent"),
):
    """Start the white agent (target being tested)."""
    start_white_agent(host=host, port=port)


@app.command()
def launch(
    env_type: str = typer.Option(
        "AlfredTWEnv",
        help="Environment type: 'AlfredTWEnv', 'AlfredThorEnv', or 'AlfredHybrid'"
    ),
    train_eval: str = typer.Option(
        "eval_out_of_distribution",
        help="Data split: 'train', 'eval_in_distribution', or 'eval_out_of_distribution'"
    ),
    num_games: int = typer.Option(
        1,
        help="Number of games to evaluate"
    ),
    max_steps: int = typer.Option(
        50,
        help="Maximum steps per game"
    ),
    task_types: Optional[str] = typer.Option(
        None,
        help="Comma-separated task type IDs (1-6), e.g., '1,2,3'. Default: all types"
    ),
):
    """Launch the complete evaluation workflow.
    
    This command:
    1. Starts both the green and white agents
    2. Sends the assessment task to the green agent
    3. Collects and displays results
    4. Terminates both agents
    """
    task_type_list = None
    if task_types:
        task_type_list = [int(t.strip()) for t in task_types.split(",")]
    
    asyncio.run(
        launch_evaluation(
            env_type=env_type,
            train_eval=train_eval,
            num_games=num_games,
            max_steps=max_steps,
            task_types=task_type_list,
        )
    )


@app.command()
def info():
    """Display information about ALFWorld task types."""
    task_types = {
        1: "pick_and_place_simple - Pick up an object and place it in a receptacle",
        2: "look_at_obj_in_light - Examine an object under a light source",
        3: "pick_clean_then_place_in_recep - Clean an object and place it somewhere",
        4: "pick_heat_then_place_in_recep - Heat an object and place it somewhere",
        5: "pick_cool_then_place_in_recep - Cool an object and place it somewhere",
        6: "pick_two_obj_and_place - Pick up two objects and place them",
    }
    
    print("\n" + "=" * 60)
    print("ALFWorld Task Types")
    print("=" * 60)
    for tid, desc in task_types.items():
        print(f"  {tid}: {desc}")
    print("=" * 60)
    print("\nUsage examples:")
    print("  python main.py launch --num-games 5")
    print("  python main.py launch --task-types '1,2' --max-steps 30")
    print("  python main.py green  # Start green agent only")
    print("  python main.py white  # Start white agent only")
    print()


if __name__ == "__main__":
    app()
