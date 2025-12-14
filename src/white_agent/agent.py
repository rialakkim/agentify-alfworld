"""White agent implementation - the target agent being tested on ALFWorld.

This is a general-purpose LLM-based agent that:
1. Receives task descriptions and observations from the green agent
2. Reasons about the task and chooses actions
3. Returns action commands to be executed in the environment

The white agent has NO knowledge of ALFWorld internals - it simply follows
the task instructions provided by the green agent, making it a truly
self-explanatory assessment following the A2A protocol.
"""

import uvicorn
import os
from dotenv import load_dotenv
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentSkill, AgentCard, AgentCapabilities
from a2a.utils import new_agent_text_message
import openai
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


SYSTEM_PROMPT = """You are an intelligent agent operating in a household environment. Your job is to complete tasks by taking appropriate actions.

IMPORTANT RULES:
1. Read the observation carefully to understand your current location and surroundings.
2. Identify your goal from the task description.
3. Choose the BEST action from the available actions list to make progress toward your goal.
4. Respond with ONLY the exact action text - no explanations, no quotes, no prefixes.

TIPS FOR SUCCESS:
- If you need to find an object, use "go to" actions to explore receptacles (desks, shelves, drawers, etc.)
- Use "take" to pick up objects, "put" to place them
- Use "open" for closed containers, "examine" or "look at" to inspect objects
- Use "use" for appliances like lamps
- Think step by step but only output the action command

Example responses (just the action, nothing else):
go to desk 1
take book 1 from desk 1
put book 1 in/on shelf 1
open drawer 1
examine book 1
use desklamp 1
"""


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
    def __init__(self, model="gpt-4o-mini", temperature=0.0):
        self.model = model
        self.temperature = temperature
        self.ctx_id_to_messages = {}
        self.client = openai.OpenAI()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        user_input = context.get_user_input()
        
        if context.context_id not in self.ctx_id_to_messages:
            self.ctx_id_to_messages[context.context_id] = [
                {"role": "system", "content": SYSTEM_PROMPT}
            ]
        
        messages = self.ctx_id_to_messages[context.context_id]
        messages.append({
            "role": "user",
            "content": user_input,
        })
        
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    messages=messages,
                    model=self.model,
                    temperature=self.temperature,
                    timeout=30,
                )
                
                next_message = response.choices[0].message
                assistant_content = next_message.content if next_message.content else ""
                
                action = assistant_content.strip()
                
                if "\n" in action:
                    action = action.split("\n")[0].strip()
                
                action = action.strip('"\'')
                
                if action.lower().startswith("action:"):
                    action = action[7:].strip()
                
                messages.append({
                    "role": "assistant",
                    "content": action,
                })
                
                await event_queue.enqueue_event(
                    new_agent_text_message(action, context_id=context.context_id)
                )
                return
                
            except Exception as e:
                last_error = e
                print(f"White agent error (attempt {attempt + 1}/{MAX_RETRIES}): {type(e).__name__}: {str(e)}")
                print(f"  Full traceback: {traceback.format_exc()}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
        
        print(f"White agent: All retries failed. Last error: {last_error}")
        admissible = self._extract_first_admissible(user_input)
        await event_queue.enqueue_event(
            new_agent_text_message(admissible, context_id=context.context_id)
        )

    def _extract_first_admissible(self, user_input: str) -> str:
        """Extract first admissible action from the observation text."""
        lines = user_input.split("\n")
        for line in lines:
            line = line.strip()
            if line and line[0].isdigit() and "." in line:
                parts = line.split(".", 1)
                if len(parts) > 1:
                    return parts[1].strip()
        return "look"

    async def cancel(self, context, event_queue) -> None:
        raise NotImplementedError


def start_white_agent(agent_name="alfworld_white_agent", host="localhost", port=9002):
    print("Starting ALFWorld white agent...")
    url = f"http://{host}:{port}"
    card = prepare_white_agent_card(url)

    model = os.environ.get("WHITE_AGENT_MODEL", "gpt-4o-mini")
    temperature = float(os.environ.get("WHITE_AGENT_TEMPERATURE", "0.0"))

    request_handler = DefaultRequestHandler(
        agent_executor=AlfWorldWhiteAgentExecutor(
            model=model,
            temperature=temperature,
        ),
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=card,
        http_handler=request_handler,
    )

    uvicorn.run(app.build(), host=host, port=port)
