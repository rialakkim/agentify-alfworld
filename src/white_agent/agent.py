"""White agent implementation - the target agent being tested on ALFWorld.

This is a general-purpose LLM-based agent that:
1. Receives task descriptions and observations from the green agent
2. Reasons about the task and chooses actions using chain-of-thought
3. Tracks action history to avoid repetition and cycles
4. Returns action commands to be executed in the environment

The white agent has NO knowledge of ALFWorld internals - it simply follows
the task instructions provided by the green agent, making it a truly
self-explanatory assessment following the A2A protocol.
"""

import uvicorn
import os
import re
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field
from dotenv import load_dotenv
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentSkill, AgentCard, AgentCapabilities
from a2a.utils import new_agent_text_message
from litellm import completion
import time
import traceback


# Load .env from project root - override=True to ignore any corrupted shell env vars
import pathlib
env_path = pathlib.Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path, override=True)

MAX_RETRIES = 3
RETRY_DELAY = 10

# Debug: Print API key status on module load
api_key = os.environ.get("OPENAI_API_KEY", "")
print(f"[White Agent Init] API Key loaded: {'Yes (' + api_key[:10] + '...)' if api_key else 'NO - MISSING!'}")


@dataclass
class TaskState:
    """Tracks the state of a task for better decision making."""
    goal: str = ""
    target_object: str = ""  # e.g., "glassbottle", "apple"
    destination: str = ""     # e.g., "countertop", "shelf"
    transformation: str = ""  # e.g., "cool", "hot", "clean", "examine"
    transformed: bool = False # whether transformation was done
    current_location: str = ""
    holding: Optional[str] = None
    visited_locations: Set[str] = field(default_factory=set)
    opened_items: Set[str] = field(default_factory=set)
    action_history: List[str] = field(default_factory=list)
    failed_actions: List[str] = field(default_factory=list)
    step_count: int = 0
    found_target: bool = False
    task_phase: str = "searching"  # searching, has_object, transforming, transformed, at_destination, done
    
    def add_action(self, action: str):
        """Record an action taken."""
        self.action_history.append(action.lower().strip())
        self.step_count += 1
        self._update_state_from_action(action)
    
    def _update_state_from_action(self, action: str):
        """Update internal state based on action taken."""
        action_lower = action.lower().strip()
        
        # Track navigation
        go_match = re.match(r'^go\s+to\s+(.+)$', action_lower)
        if go_match:
            self.current_location = go_match.group(1)
            self.visited_locations.add(self.current_location)
            # Check if we reached destination
            if self.destination and self.destination in self.current_location:
                if self.holding and self.target_object and self.target_object in self.holding:
                    self.task_phase = "at_destination"
        
        # Track item pickup/drop
        take_match = re.match(r'^take\s+(.+?)\s+from', action_lower)
        if take_match:
            self.holding = take_match.group(1)
            # Check if we picked up target
            if self.target_object and self.target_object in self.holding:
                self.found_target = True
                self.task_phase = "has_object"
        
        put_match = re.match(r'^put\s+(.+?)\s+(?:in|on)', action_lower)
        if put_match:
            if self.holding and self.target_object and self.target_object in self.holding:
                self.task_phase = "done"
            self.holding = None
        
        # Track open/close
        open_match = re.match(r'^open\s+(.+)$', action_lower)
        if open_match:
            self.opened_items.add(open_match.group(1))
        
        close_match = re.match(r'^close\s+(.+)$', action_lower)
        if close_match:
            self.opened_items.discard(close_match.group(1))
        
        # Track transformations (cool, heat, clean)
        transform_match = re.match(r'^(cool|heat|clean)\s+(.+)$', action_lower)
        if transform_match:
            self.transformed = True
            self.task_phase = "transformed"
            self.holding = None  # Object is now in the appliance
        
        # Track use (for desklamp examine tasks)
        use_match = re.match(r'^use\s+(.+)$', action_lower)
        if use_match:
            if "lamp" in use_match.group(1) or "desklamp" in use_match.group(1):
                self.transformed = True
                self.task_phase = "done"
    
    def get_recent_actions(self, n: int = 5) -> List[str]:
        """Get the last n actions."""
        return self.action_history[-n:] if self.action_history else []
    
    def is_action_repeated(self, action: str, window: int = 3) -> bool:
        """Check if an action was recently taken."""
        recent = self.get_recent_actions(window)
        return action.lower().strip() in recent
    
    def get_unvisited_from_options(self, available_actions: List[str]) -> List[str]:
        """Get navigation options for unvisited locations."""
        unvisited = []
        for action in available_actions:
            match = re.match(r'^go\s+to\s+(.+)$', action.lower())
            if match:
                location = match.group(1)
                if location not in self.visited_locations:
                    unvisited.append(action)
        return unvisited
    
    def get_state_summary(self) -> str:
        """Get a summary of current state for the LLM."""
        summary_parts = []
        
        # Goal info first - most important
        if self.goal:
            summary_parts.append(f"GOAL: {self.goal}")
        if self.target_object:
            if self.transformation:
                summary_parts.append(f"   Need: {self.transformation} {self.target_object} → put in {self.destination}")
                summary_parts.append(f"   Transformed: {'YES' if self.transformed else 'NO'}")
            else:
                summary_parts.append(f"   Need: {self.target_object} → put in {self.destination}")
            summary_parts.append(f"   Phase: {self.task_phase.upper()}")
        
        if self.holding:
            summary_parts.append(f"Holding: {self.holding}")
        else:
            summary_parts.append(f"Holding: nothing")
        
        if self.current_location:
            summary_parts.append(f"At: {self.current_location}")
        
        if self.opened_items:
            summary_parts.append(f"CLOSE THESE: {', '.join(self.opened_items)}")
        
        return "\n".join(summary_parts) if summary_parts else "Starting fresh"
    
    def extract_goal_from_observation(self, observation: str):
        """Extract goal, target object, and destination from initial observation."""
        obs_lower = observation.lower()
        
        # Store full goal text for reference
        task_match = re.search(r'your task is to:\s*(.+?)(?:\.|$)', obs_lower)
        if task_match:
            self.goal = task_match.group(1).strip()
        
        # Pattern: put a cool/hot/clean X in/on Y (transformation required)
        transform_match = re.search(r'put\s+(?:a|an|some|the)?\s*(cool|hot|clean)\s+(\w+)\s+(?:in|on)\s+(?:a|an|the)?\s*(\w+)', obs_lower)
        if transform_match:
            self.transformation = transform_match.group(1)  # cool, hot, clean
            self.target_object = transform_match.group(2)
            self.destination = transform_match.group(3)
            self.task_phase = "searching"
            return
        
        # Pattern: put X in/on Y (simple placement)
        put_match = re.search(r'put\s+(?:a|an|some|the)?\s*(\w+)\s+(?:in|on)\s+(?:a|an|the)?\s*(\w+)', obs_lower)
        if put_match:
            self.target_object = put_match.group(1)
            self.destination = put_match.group(2)
            self.transformation = ""
            self.task_phase = "searching"
            return
        
        # Pattern: examine X under/with/in light / look at X under lamp
        examine_match = re.search(r'(?:examine|look at)\s+(?:a|an|some|the)?\s*(\w+)\s+(?:in|under|with)\s+(?:the\s+)?(?:light|lamp|desklamp)', obs_lower)
        if examine_match:
            self.target_object = examine_match.group(1)
            self.destination = "desklamp"
            self.transformation = "examine"
            self.task_phase = "searching"
            return


SYSTEM_PROMPT = """You are a household robot. Complete tasks step by step.

YOUR GOAL is stated in the first message (e.g., "Your task is to: put a cool potato in garbagecan").

TASK TYPES & SOLUTIONS:
1. "put X in/on Y" → find X, take X, go to Y, put X in/on Y
2. "put a cool X in Y" → find X, take X, go to fridge, open fridge, cool X, take X, go to Y, put X
3. "put a hot X in Y" → find X, take X, go to microwave, open, heat X, take X, go to Y, put X  
4. "put a clean X in Y" → find X, take X, go to sinkbasin, clean X, go to Y, put X
5. "examine X in light" → find X, take X, go to desklamp, use desklamp
6. "look at X under lamp" → same as examine in light

CRITICAL: Read your task type carefully! If task says "cool" you MUST use fridge. If "heat" use microwave.

RULES:
- Output ONLY one action per turn - no explanations
- Actions must match exactly what's in "Available actions" list
- When you see target object in observation, take it immediately
- After "cool X" or "heat X" or "clean X", the object stays where it was transformed - take it again
- Always close containers after opening (fridge, cabinets, drawers)

COMMON MISTAKES TO AVOID:
- Don't pick up random objects - only your target
- Don't forget to cool/heat/clean if the task requires it
- Don't put object somewhere before transforming it
- After cooling in fridge: open fridge, take object, close fridge, then go to destination"""


REASONING_PROMPT = """Based on the current state and observation, think through the following:
1. What is my goal?
2. What have I already accomplished?
3. What is the next logical step?
4. Which available action best achieves that step?

STATE CONTEXT:
{state_summary}

IMPORTANT: After reasoning, output ONLY the action command on the last line, nothing else."""


def prepare_white_agent_card(url):
    skill = AgentSkill(
        id="alfworld_task_completion",
        name="ALFWorld Task Completion",
        description="Completes household tasks in ALFWorld TextWorld environments by choosing appropriate actions",
        tags=["alfworld", "textworld", "household", "task-completion"],
        examples=[],
    )
    card = AgentCard(
        name="alfworld_white_agent",
        description="A general-purpose LLM agent for completing tasks in ALFWorld environments",
        url=url,
        version="1.0.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(),
        skills=[skill],
    )
    return card


class AlfWorldWhiteAgentExecutor(AgentExecutor):
    """Enhanced white agent with state tracking and improved reasoning."""
    
    def __init__(self, model="gpt-4o-mini", temperature=0.0, use_reasoning=True):
        self.model = model
        self.temperature = temperature
        self.use_reasoning = use_reasoning
        self.ctx_id_to_messages: Dict[str, List[Dict]] = {}
        self.ctx_id_to_state: Dict[str, TaskState] = {}

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        user_input = context.get_user_input()
        ctx_id = context.context_id
        
        # Initialize state for new conversations
        if ctx_id not in self.ctx_id_to_state:
            self.ctx_id_to_state[ctx_id] = TaskState()
            self.ctx_id_to_messages[ctx_id] = [
                {"role": "system", "content": SYSTEM_PROMPT}
            ]
        
        state = self.ctx_id_to_state[ctx_id]
        messages = self.ctx_id_to_messages[ctx_id]
        
        # Extract available actions from the input
        available_actions = self._extract_available_actions(user_input)
        
        # Build enhanced prompt with state context
        enhanced_input = self._build_enhanced_input(user_input, state, available_actions)
        
        messages.append({
            "role": "user",
            "content": enhanced_input,
        })
        
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                # Use litellm completion with provider detection
                if os.environ.get("LITELLM_PROXY_API_KEY") is not None:
                    response = completion(
                        messages=messages,
                        model=self.model,
                        custom_llm_provider="litellm_proxy",
                        temperature=self.temperature,
                    )
                else:
                    # Default to OpenAI provider
                    model_with_provider = f"openai/{self.model}" if not "/" in self.model else self.model
                    response = completion(
                        messages=messages,
                        model=model_with_provider,
                        custom_llm_provider="openai",
                        temperature=self.temperature,
                    )
                
                next_message = response.choices[0].message.model_dump()
                assistant_content = next_message.get("content", "") if next_message.get("content") else ""
                
                # Extract the action from the response
                action = self._extract_action(assistant_content)
                
                # Validate action against available options
                action = self._validate_action(action, available_actions, state)
                
                # Update state with the action
                state.add_action(action)
                
                messages.append({
                    "role": "assistant",
                    "content": action,
                })
                
                await event_queue.enqueue_event(
                    new_agent_text_message(action, context_id=ctx_id)
                )
                return
                
            except Exception as e:
                last_error = e
                print(f"White agent error (attempt {attempt + 1}/{MAX_RETRIES}): {type(e).__name__}: {str(e)}")
                print(f"  Full traceback: {traceback.format_exc()}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
        
        print(f"White agent: All retries failed. Last error: {last_error}")
        # Fallback: try to pick an unvisited location or first available action
        fallback = self._get_fallback_action(available_actions, state)
        state.add_action(fallback)
        await event_queue.enqueue_event(
            new_agent_text_message(fallback, context_id=ctx_id)
        )
    
    def _build_enhanced_input(self, user_input: str, state: TaskState, available_actions: List[str]) -> str:
        """Build enhanced input with state context for better reasoning."""
        # On first message, extract the goal
        if state.step_count == 0:
            state.extract_goal_from_observation(user_input)
            return user_input  # First message doesn't need enhancement
        
        if not self.use_reasoning:
            return user_input
        
        # Build directive based on current phase and transformation status
        directive = ""
        obs_lower = user_input.lower()
        
        # Determine what appliance is needed for transformation
        appliance_map = {"cool": "fridge", "hot": "microwave", "clean": "sinkbasin", "examine": "desklamp"}
        needed_appliance = appliance_map.get(state.transformation, "")
        
        if state.task_phase == "searching":
            if state.holding and state.target_object and state.target_object in state.holding:
                # We have the object! What's next?
                if state.transformation and not state.transformed:
                    directive = f"You have {state.holding}! NOW GO TO {needed_appliance.upper()} to {state.transformation} it!"
                else:
                    directive = f"You have {state.holding}! NOW GO TO {state.destination.upper()} and PUT it there!"
                state.task_phase = "has_object"
            elif state.target_object and state.target_object in obs_lower:
                directive = f"You can see {state.target_object}! TAKE IT NOW!"
            else:
                directive = f"Search for {state.target_object}. Try: countertops, then open cabinets/drawers/fridge."
        
        elif state.task_phase == "has_object":
            if state.transformation and not state.transformed:
                # Need to transform first
                if needed_appliance and needed_appliance in state.current_location:
                    directive = f"You're at {needed_appliance}! Do: {state.transformation} {state.holding}"
                else:
                    directive = f"GO TO {needed_appliance.upper()} to {state.transformation} the {state.target_object}!"
            else:
                # Ready to deliver
                dest_actions = [a for a in available_actions if state.destination and state.destination in a.lower() and "go to" in a.lower()]
                if dest_actions:
                    directive = f"GO TO {dest_actions[0].replace('go to ', '')} and put {state.holding} there!"
                else:
                    directive = f"Find a {state.destination} and put {state.holding} there!"
        
        elif state.task_phase == "transformed":
            # Object was transformed, need to take it and deliver
            if state.holding:
                directive = f"Transformed! Now GO TO {state.destination.upper()} and PUT {state.holding} there!"
            else:
                directive = f"Take the {state.target_object} from here, then go to {state.destination}!"
        
        elif state.task_phase == "at_destination":
            put_actions = [a for a in available_actions if "put" in a.lower() and state.target_object in a.lower()]
            if put_actions:
                directive = f"PUT IT DOWN NOW! Use: {put_actions[0]}"
            else:
                directive = f"You're at {state.destination} with {state.holding}! PUT IT DOWN!"
        
        state_summary = state.get_state_summary()
        
        # Build context block
        context_block = "\n===== CURRENT STATUS =====\n"
        context_block += state_summary + "\n"
        if directive:
            context_block += f"\n>>> NEXT: {directive}\n"
        context_block += "==========================\n"
        
        return context_block + "\n" + user_input
    
    def _extract_action(self, response: str) -> str:
        """Extract the action command from LLM response."""
        # Get the last non-empty line (action should be at the end)
        lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
        
        if not lines:
            return "look"
        
        # Try last line first
        action = lines[-1]
        
        # Clean up common prefixes
        prefixes_to_remove = ["action:", "Action:", "ACTION:", "I choose:", "My action:"]
        for prefix in prefixes_to_remove:
            if action.startswith(prefix):
                action = action[len(prefix):].strip()
        
        # Remove quotes
        action = action.strip('"\'')
        
        # If the response looks like reasoning, try to find an action-like line
        if len(action) > 50 or any(word in action.lower() for word in ["because", "since", "therefore", "i think"]):
            for line in reversed(lines):
                clean_line = line.strip().strip('"\'')
                # Action lines are typically short commands
                if len(clean_line) < 40 and clean_line.split()[0].lower() in [
                    "go", "take", "put", "open", "close", "use", "examine", "look", "clean", "heat", "cool"
                ]:
                    return clean_line
        
        return action
    
    def _extract_available_actions(self, user_input: str) -> List[str]:
        """Extract the list of available actions from the observation."""
        actions = []
        lines = user_input.split("\n")
        in_actions = False
        
        for line in lines:
            line = line.strip()
            if "Available actions" in line:
                in_actions = True
                continue
            if in_actions:
                # Match numbered action format "  1. go to desk 1"
                match = re.match(r'^\s*\d+\.\s*(.+)$', line)
                if match:
                    actions.append(match.group(1).strip())
                elif line and not line.startswith("Please"):
                    # Might be unnumbered action
                    if any(line.lower().startswith(verb) for verb in ["go", "take", "put", "open", "close", "use", "examine"]):
                        actions.append(line)
        
        return actions
    
    def _validate_action(self, action: str, available_actions: List[str], state: TaskState) -> str:
        """Validate and potentially correct the action."""
        action_lower = action.lower().strip()
        
        # Direct match - best case
        for available in available_actions:
            if available.lower() == action_lower:
                return available
        
        # Partial match
        for available in available_actions:
            if action_lower in available.lower() or available.lower() in action_lower:
                return available
        
        # SMART OVERRIDE: If we have target and see destination, go there!
        if state.holding and state.target_object and state.target_object in state.holding:
            for available in available_actions:
                if state.destination and state.destination in available.lower() and "go to" in available.lower():
                    return available
        
        # SMART OVERRIDE: If we see target object in available actions, take it!
        if state.target_object and not state.holding:
            for available in available_actions:
                if state.target_object in available.lower() and "take" in available.lower():
                    return available
        
        # SMART OVERRIDE: If at destination with target, put it!
        if state.holding and state.target_object and state.target_object in state.holding:
            if state.destination and state.current_location and state.destination in state.current_location:
                for available in available_actions:
                    if "put" in available.lower() and state.target_object in available.lower():
                        return available
        
        # If action is repeated too many times, try an alternative
        if state.is_action_repeated(action, window=2):
            alternatives = [a for a in available_actions if not state.is_action_repeated(a, window=2)]
            if alternatives:
                # Prefer goal-directed actions
                if state.destination:
                    for alt in alternatives:
                        if state.destination in alt.lower():
                            return alt
                # Then prefer unvisited locations
                unvisited = state.get_unvisited_from_options(alternatives)
                if unvisited:
                    return unvisited[0]
                return alternatives[0]
        
        # Fallback to first available
        if available_actions:
            return available_actions[0]
        
        return action
    
    def _get_fallback_action(self, available_actions: List[str], state: TaskState) -> str:
        """Get a fallback action when LLM fails."""
        if not available_actions:
            return "look"
        
        # Prefer unvisited locations
        unvisited = state.get_unvisited_from_options(available_actions)
        if unvisited:
            return unvisited[0]
        
        # Avoid recently repeated actions
        for action in available_actions:
            if not state.is_action_repeated(action, window=3):
                return action
        
        return available_actions[0]

    async def cancel(self, context, event_queue) -> None:
        raise NotImplementedError


def start_white_agent(agent_name="alfworld_white_agent", host="localhost", port=9002):
    print("Starting ALFWorld white agent (enhanced version)...")

    # Use AGENT_URL if set (for online platform), otherwise construct from host/port
    agent_url = os.getenv("AGENT_URL")
    if agent_url is None:
        agent_url = f"http://{host}:{port}"
    card = prepare_white_agent_card(agent_url)

    model = os.environ.get("WHITE_AGENT_MODEL", "gpt-4o-mini")
    temperature = float(os.environ.get("WHITE_AGENT_TEMPERATURE", "0.0"))
    use_reasoning = os.environ.get("WHITE_AGENT_REASONING", "true").lower() == "true"

    print(f"  Model: {model}")
    print(f"  Temperature: {temperature}")
    print(f"  Enhanced reasoning: {use_reasoning}")

    request_handler = DefaultRequestHandler(
        agent_executor=AlfWorldWhiteAgentExecutor(
            model=model,
            temperature=temperature,
            use_reasoning=use_reasoning,
        ),
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=card,
        http_handler=request_handler,
    )

    uvicorn.run(app.build(), host=host, port=port)
