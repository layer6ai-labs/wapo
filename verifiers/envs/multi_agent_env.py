import logging
import re
from typing import Optional

from environments.multi_agent.agents.answer_agent import AnswerAgent
from environments.multi_agent.agents.multi_agent_config import (
    AnswerAgentConfig,
    MultiAgentConfig,
)
from verifiers.envs.multiturn_env import MultiTurnEnv
from verifiers.types import (
    ChatMessage,
    Info,
    Messages,
    State,
)

logger = logging.getLogger(__name__)


class MultiAgentEnv(MultiTurnEnv):
    """
    Multi-agent environment where a Planning Agent (trained via RL)
    interacts with an Answer Agent (external, not trained).

    The Planning Agent generates queries using <query> tags.
    The Answer Agent retrieves relevant documents and responds with answers.
    The environment formats responses as <user_answer> for the Planning Agent.

    Flow:
        Planning Agent → <query>question</query>
               ↓
        Retriever fetches docs (configurable URL, top_k)
               ↓
        Format: <query>question</query>\\n<context>doc1\\n\\ndoc2</context>
               ↓
        Answer Agent LLM generates answer based on context
               ↓
        Return <user_answer>answer</user_answer> to Planning Agent
    """

    def __init__(
        self,
        answer_agent_config: Optional[AnswerAgentConfig] = None,
        multi_agent_config: Optional[MultiAgentConfig] = None,
        **kwargs,
    ):
        """
        Initialize the MultiAgentEnv.

        Args:
            answer_agent_config: Configuration for the answer agent.
                If provided, overrides multi_agent_config.answer_agent.
            multi_agent_config: Full multi-agent configuration.
            **kwargs: Additional arguments passed to MultiTurnEnv.
        """
        # Use provided config or defaults
        self.multi_agent_config = multi_agent_config or MultiAgentConfig()
        if answer_agent_config is not None:
            self.multi_agent_config.answer_agent = answer_agent_config

        # Initialize answer agent
        self.answer_agent = AnswerAgent(self.multi_agent_config.answer_agent)

        # Tag configuration
        self.query_tag = self.multi_agent_config.query_tag
        self.answer_tag = self.multi_agent_config.answer_tag
        self.user_answer_tag = self.multi_agent_config.user_answer_tag

        # Get system prompt from config if not already in kwargs
        if "system_prompt" not in kwargs:
            kwargs["system_prompt"] = self.multi_agent_config.planning_system_prompt

        # Call parent init with max_turns from config
        super().__init__(
            max_turns=self.multi_agent_config.max_turns,
            **kwargs,
        )

    def _extract_tag_content(self, text: str, tag: str) -> Optional[str]:
        """Extract content between XML-like tags."""
        pattern = rf"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def _has_final_answer(self, messages: Messages) -> bool:
        """Check if the last assistant message contains a final answer tag."""
        if not isinstance(messages, list) or not messages:
            return False

        # Look for the last assistant message
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                return (
                    f"<{self.answer_tag}>" in content
                    and f"</{self.answer_tag}>" in content
                )
        return False

    def _has_query(self, text: str) -> bool:
        """Check if the text contains a query tag."""
        return f"<{self.query_tag}>" in text and f"</{self.query_tag}>" in text

    async def is_completed(self, messages: Messages, state: State, **kwargs) -> bool:
        """
        Check if the interaction is complete.

        Complete when:
        - Max turns reached
        - Planning agent provides final answer (<answer> tag)
        - Prompt too long
        """
        max_turns = await self.max_turns_reached(state)
        prompt_too_long = await self.prompt_too_long(state)
        has_final = self._has_final_answer(messages)

        return max_turns or prompt_too_long or has_final

    async def env_response(
        self, messages: Messages, state: State, **kwargs
    ) -> tuple[Messages, State]:
        """
        Generate environment response using the Answer Agent.

        When the Planning Agent emits a <query>, we:
        1. Extract the query
        2. Call the Answer Agent (which retrieves docs and generates answer)
        3. Format response as <user_answer>

        Args:
            messages: Current conversation messages.
            state: Current state dictionary.
            **kwargs: Additional arguments.

        Returns:
            Tuple of (new messages, updated state).
        """
        assert isinstance(messages, list)

        # Get the last assistant message
        last_assistant_content = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                last_assistant_content = msg.get("content", "")
                break

        # Extract query from planning agent's response
        query = self._extract_tag_content(last_assistant_content, self.query_tag)

        if query is None:
            # No query found - prompt for proper format
            env_message: ChatMessage = {
                "role": "user",
                "content": (
                    f"Invalid format. Please use <{self.query_tag}>your query</{self.query_tag}> "
                    f"to ask questions, or <{self.answer_tag}>final answer</{self.answer_tag}> "
                    "to provide your final answer."
                ),
            }
            return [env_message], state

        # Call the answer agent (retrieves docs + generates answer)
        answer = await self.answer_agent.answer(query)

        # Track the interaction in state
        if "agent_interactions" not in state:
            state["agent_interactions"] = []
        state["agent_interactions"].append(
            {
                "turn": state["turn"],
                "query": query,
                "answer": answer,
            }
        )

        # Format response for planning agent
        # Return answer directly without wrapping tags
        env_message: ChatMessage = {
            "role": "user",
            "content": f"<{self.user_answer_tag}>{answer}</{self.user_answer_tag}>",
            # "content": answer,
        }

        return [env_message], state

    async def init_state(
        self,
        prompt: Messages,
        completion: Messages,
        answer: str,
        task: str,
        info: Info,
        example_id: int,
        **kwargs,
    ) -> State:
        """Initialize state with multi-agent specific fields."""
        state = await super().init_state(
            prompt, completion, answer, task, info, example_id, **kwargs
        )
        # Add multi-agent specific fields
        state["agent_interactions"] = []
        state["final_answer"] = ""
        return state

    async def setup_state(self, state: State, **kwargs) -> State:
        """Setup state before rollout begins."""
        state = await super().setup_state(state, **kwargs)
        # Reset multi-agent fields if needed
        if "agent_interactions" not in state:
            state["agent_interactions"] = []
        if "final_answer" not in state:
            state["final_answer"] = ""
        return state

    def get_final_answer(self, state: State) -> Optional[str]:
        """
        Extract the final answer from the state after rollout.

        Args:
            state: State dictionary after rollout completion.

        Returns:
            The final answer if found, None otherwise.
        """
        if state.get("final_answer"):
            return state["final_answer"]

        # Try to extract from completion
        completion = state.get("completion", [])
        if isinstance(completion, list):
            for msg in reversed(completion):
                if msg.get("role") == "assistant":
                    final = self._extract_tag_content(
                        msg.get("content", ""), self.answer_tag
                    )
                    if final:
                        state["final_answer"] = final
                        return final
        return None

    async def close(self):
        """Close resources."""
        await self.answer_agent.close()
