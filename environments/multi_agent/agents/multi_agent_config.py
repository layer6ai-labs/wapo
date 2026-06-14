from dataclasses import dataclass, field
from typing import List


@dataclass
class RetrieverConfig:
    """Configuration for the retrieval service."""

    url: str = "http://localhost:8000/retrieve"
    """URL of the retrieval server endpoint."""

    top_k: int = 3
    """Number of documents to retrieve."""

    return_scores: bool = False
    """Whether to return relevance scores with documents."""

    timeout: float = 30.0
    """Request timeout in seconds."""


@dataclass
class AnswerAgentConfig:
    """Configuration for the Answer Agent (external endpoint)."""

    # Endpoint configuration
    base_url: str = "http://localhost:8001/v1"
    """Base URL for the answer agent LLM endpoint."""

    api_key: str = "EMPTY"
    """API key for the answer agent endpoint."""

    model_name: str = "Qwen/Qwen3-8B"
    """Model name to use for the answer agent."""

    # Connection settings
    timeout: float = 120.0
    """Request timeout in seconds."""

    max_connections: int = 100
    """Maximum number of concurrent connections."""

    max_retries: int = 3
    """Maximum number of retries for failed requests."""

    # Generation settings
    max_tokens: int = 2048
    """Maximum tokens for answer generation."""

    temperature: float = 0.7
    """Sampling temperature."""

    top_p: float = 0.95
    """Top-p (nucleus) sampling parameter."""

    stop_words: List[str] = field(default_factory=lambda: ["</answer>"])
    """Stop words for generation."""

    # Retriever configuration
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    """Configuration for the retrieval service."""

    # System prompt
    system_prompt: str = field(
        default=(
            "You are an expert in answering questions based on provided context. "
            "Do not include any additional information that is not present in the context. "
            "Your task is to generate a concise and accurate answer to the user's query "
            "using the context provided. Ensure that your answer is directly relevant to "
            "the question and is supported by the context. "
            "Your response should be formatted as follows: <answer>your_answer_here</answer> "
            "If the context does not provide sufficient information, indicate that the answer "
            "is not available in the following format: "
            "<answer>Not available in the provided context</answer>"
        )
    )
    """System prompt for the answer agent."""


@dataclass
class MultiAgentConfig:
    """Configuration for multi-agent setup."""

    answer_agent: AnswerAgentConfig = field(default_factory=AnswerAgentConfig)
    """Configuration for the answer agent."""

    max_turns: int = 10
    """Maximum number of turns for multi-turn interaction."""

    query_tag: str = "query"
    """XML tag for queries from the planning agent."""

    answer_tag: str = "answer"
    """XML tag for final answers from the planning agent."""

    user_answer_tag: str = "user_answer"
    """XML tag for answers returned to the planning agent."""

    planning_system_prompt: str = field(
        default=(
            "You are a planning agent tasked to answer multi-hop questions. "
            "Generate queries one at a time using <query>your_query</query>. "
            "The user will respond with <user_answer>answer</user_answer>. "
            "When you have gathered enough information, provide your final answer "
            "using <answer>final_answer</answer>."
        )
    )
    """System prompt for the planning agent."""
