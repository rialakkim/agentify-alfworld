"""Repetitive white agent - selects first available action for testing."""

import uvicorn
import dotenv
import os
import re
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentSkill, AgentCard, AgentCapabilities
from a2a.utils import new_agent_text_message


dotenv.load_dotenv()


def prepare_repetitive_white_agent_card(url):
    """Prepare agent card for repetitive white agent."""
    skill = AgentSkill(
        id="task_fulfillment",
        name="Task Fulfillment",
        description="Selects the first available action at each step",
        tags=["general", "test"],
        examples=[],
    )
    card = AgentCard(
        name="repetitive_white_agent",
        description="A white agent that always picks the first available action for testing behavior metrics.",
        url=url,
        version="1.0.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(),
        skills=[skill],
    )
    return card


class RepetitiveWhiteAgentExecutor(AgentExecutor):
    """Agent executor that always selects the first available action."""
    
    def __init__(self):
        self.ctx_id_to_messages = {}
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute by selecting the first available action."""
        user_input = context.get_user_input()
        context_id = context.context_id
        
        if context_id not in self.ctx_id_to_messages:
            self.ctx_id_to_messages[context_id] = []
        
        messages = self.ctx_id_to_messages[context_id]
        
        # Extract first available action
        response_text = self._extract_first_action(user_input)
        
        messages.append({
            "role": "user",
            "content": user_input,
        })
        messages.append({
            "role": "assistant",
            "content": response_text,
        })
        
        await event_queue.enqueue_event(
            new_agent_text_message(response_text, context_id=context_id)
        )
    
    def _extract_first_action(self, user_input: str) -> str:
        """Extract and return the first available action from the formatted action list."""
        # Look for the pattern "  1. action_text" in the input
        match = re.search(r'^\s*1\.\s*(.+)$', user_input, re.MULTILINE)
        
        if match:
            action = match.group(1).strip()
            return action
        
        # Fallback: return a default action if no actions found
        return "go to kitchen"
    
    async def cancel(self, context, event_queue) -> None:
        """Cancel execution."""
        raise NotImplementedError


def start_repetitive_white_agent(
    agent_name="repetitive_white_agent",
    host="localhost",
    port=9003
):
    """Start the repetitive white agent server."""
    print("Starting repetitive white agent...")
    
    # Use AGENT_URL if set (for online platform), otherwise construct from host/port
    agent_url = os.getenv("AGENT_URL")
    if agent_url is None:
        agent_url = f"http://{host}:{port}"
    
    card = prepare_repetitive_white_agent_card(agent_url)
    
    request_handler = DefaultRequestHandler(
        agent_executor=RepetitiveWhiteAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )
    
    app = A2AStarletteApplication(
        agent_card=card,
        http_handler=request_handler,
    )
    
    uvicorn.run(app.build(), host=host, port=port)


if __name__ == "__main__":
    start_repetitive_white_agent()
