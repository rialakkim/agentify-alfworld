"""Alternating white agent - alternates between first and second available actions."""

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


def prepare_alternating_white_agent_card(url):
    """Prepare agent card for alternating white agent."""
    skill = AgentSkill(
        id="task_fulfillment",
        name="Task Fulfillment",
        description="Alternates between the first and second available actions",
        tags=["general", "test"],
        examples=[],
    )
    card = AgentCard(
        name="alternating_white_agent",
        description="A white agent that alternates between the first and second available actions for testing behavior metrics.",
        url=url,
        version="1.0.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(),
        skills=[skill],
    )
    return card


class AlternatingWhiteAgentExecutor(AgentExecutor):
    """Agent executor that alternates between first and second available actions."""
    
    def __init__(self):
        self.ctx_id_to_messages = {}
        self.ctx_id_to_step_count = {}
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute by alternating between first and second available actions."""
        user_input = context.get_user_input()
        context_id = context.context_id
        
        if context_id not in self.ctx_id_to_messages:
            self.ctx_id_to_messages[context_id] = []
            self.ctx_id_to_step_count[context_id] = 0
        
        messages = self.ctx_id_to_messages[context_id]
        step_count = self.ctx_id_to_step_count[context_id]
        
        # Extract alternating action
        response_text = self._extract_alternating_action(user_input, step_count)
        
        messages.append({
            "role": "user",
            "content": user_input,
        })
        messages.append({
            "role": "assistant",
            "content": response_text,
        })
        
        self.ctx_id_to_step_count[context_id] += 1
        
        await event_queue.enqueue_event(
            new_agent_text_message(response_text, context_id=context_id)
        )
    
    def _extract_alternating_action(self, user_input: str, step_count: int) -> str:
        """Extract and return alternating between 1st and 2nd available actions."""
        # Look for the pattern "  1. action_text" and "  2. action_text"
        match_1 = re.search(r'^\s*1\.\s*(.+)$', user_input, re.MULTILINE)
        match_2 = re.search(r'^\s*2\.\s*(.+)$', user_input, re.MULTILINE)
        
        # Alternate: even steps use 1st action, odd steps use 2nd action
        if step_count % 2 == 0:
            # Use first action
            if match_1:
                action = match_1.group(1).strip()
                return action
        else:
            # Use second action
            if match_2:
                action = match_2.group(1).strip()
                return action
        
        # Fallback: return first action if not enough actions
        if match_1:
            return match_1.group(1).strip()
        
        return "go to kitchen"
    
    async def cancel(self, context, event_queue) -> None:
        """Cancel execution."""
        raise NotImplementedError


def start_alternating_white_agent(
    agent_name="alternating_white_agent",
    host="localhost",
    port=9003
):
    """Start the alternating white agent server."""
    print("Starting alternating white agent...")
    
    # Use AGENT_URL if set (for online platform), otherwise construct from host/port
    agent_url = os.getenv("AGENT_URL")
    if agent_url is None:
        agent_url = f"http://{host}:{port}"
    
    card = prepare_alternating_white_agent_card(agent_url)
    
    request_handler = DefaultRequestHandler(
        agent_executor=AlternatingWhiteAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )
    
    app = A2AStarletteApplication(
        agent_card=card,
        http_handler=request_handler,
    )
    
    uvicorn.run(app.build(), host=host, port=port)


if __name__ == "__main__":
    start_alternating_white_agent()
