import asyncio
import logging
import re
from typing import List, Optional

import httpx
from openai import AsyncOpenAI

from .multi_agent_config import AnswerAgentConfig

logger = logging.getLogger(__name__)


class AnswerAgent:
    """
    Async client for the Answer Agent that handles:
    1. Retrieving relevant documents from a retrieval service
    2. Generating answers based on retrieved context using an external LLM

    The answer agent is NOT trained - it's a fixed external endpoint.
    """

    def __init__(self, config: AnswerAgentConfig):
        """
        Initialize the AnswerAgent.

        Args:
            config: Configuration for the answer agent including retriever settings.
        """
        self.config = config
        self._llm_client: Optional[AsyncOpenAI] = None
        self._http_client: Optional[httpx.AsyncClient] = None

    async def _get_llm_client(self) -> AsyncOpenAI:
        """Lazy initialization of AsyncOpenAI client for LLM calls."""
        if self._llm_client is None:
            self._llm_client = AsyncOpenAI(
                base_url=self.config.base_url,
                api_key=self.config.api_key,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
            )
        return self._llm_client

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Lazy initialization of HTTP client for retriever calls."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                limits=httpx.Limits(max_connections=self.config.max_connections),
                timeout=httpx.Timeout(self.config.retriever.timeout),
            )
        return self._http_client

    async def retrieve(self, query: str) -> List[str]:
        """
        Call the retriever endpoint to get relevant documents.

        Args:
            query: The query to search for.

        Returns:
            List of document contents.
        """
        http_client = await self._get_http_client()

        payload = {
            "queries": [query],
            "top_k": self.config.retriever.top_k,
            "search_mode": "dense",
        }

        try:
            response = await http_client.post(
                self.config.retriever.url,
                json=payload,
            )
            response.raise_for_status()
            result = response.json()

            # Extract document contents from the response
            # Expected format: [[str]]
            docs = result[0]
            # Expected format: {"result": [[{"document": {"contents": "..."}, "score": ...}, ...]]}
            # if "result" in result and result["result"]:
            #     for doc_item in result["result"][0]:
            #         if isinstance(doc_item, dict) and "document" in doc_item:
            #             doc = doc_item["document"]
            #             # Handle both "contents" and "text" keys
            #             content = doc.get("contents") or doc.get("text", "")
            #             if content:
            #                 docs.append(content)
            return docs

        except httpx.HTTPError as e:
            logger.error(f"Retriever request failed: {e}")
            return []
        except (KeyError, TypeError, IndexError) as e:
            logger.error(f"Failed to parse retriever response: {e}")
            return []

    async def retrieve_batch(
        self, queries: List[str], max_concurrent: int = 10
    ) -> List[List[str]]:
        """
        Retrieve documents for a batch of queries concurrently.

        Args:
            queries: List of queries to search for.
            max_concurrent: Maximum concurrent requests.

        Returns:
            List of document lists, one per query.
        """
        http_client = await self._get_http_client()

        # Use batch endpoint if available
        payload = {
            "queries": queries,
            "topk": self.config.retriever.top_k,
            "return_scores": self.config.retriever.return_scores,
        }

        try:
            response = await http_client.post(
                self.config.retriever.url,
                json=payload,
            )
            response.raise_for_status()
            result = response.json()

            # Parse batch results
            all_docs = []
            if "result" in result:
                for query_result in result["result"]:
                    docs = []
                    for doc_item in query_result:
                        if isinstance(doc_item, dict) and "document" in doc_item:
                            doc = doc_item["document"]
                            content = doc.get("contents") or doc.get("text", "")
                            if content:
                                docs.append(content)
                    all_docs.append(docs)
            return all_docs

        except (httpx.HTTPError, KeyError, TypeError) as e:
            logger.error(f"Batch retriever request failed: {e}")
            # Fall back to individual requests
            semaphore = asyncio.Semaphore(max_concurrent)

            async def retrieve_with_sem(query: str) -> List[str]:
                async with semaphore:
                    return await self.retrieve(query)

            return await asyncio.gather(*[retrieve_with_sem(q) for q in queries])

    def _format_context(self, docs: List[str]) -> str:
        """Format retrieved documents as context."""
        if not docs:
            return "<context>No relevant documents found.</context>"
        return "<context>" + "\n\n".join(docs) + "</context>"

    def _extract_answer(self, text: str) -> str:
        """Extract content between <answer> tags."""
        pattern = r"<answer>(.*?)</answer>"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        # If no tags found, return the full text stripped
        return text.strip()

    async def answer(self, query: str) -> str:
        """
        Get an answer for a query by:
        1. Retrieving relevant documents
        2. Calling the answer agent LLM with the context

        Args:
            query: The query to answer.

        Returns:
            The extracted answer text.
        """
        # Step 1: Retrieve relevant documents
        docs = await self.retrieve(query)
        context = self._format_context(docs)

        # Step 2: Format input for answer agent
        formatted_input = f"<query>{query}</query>\n{context}"

        # Step 3: Call answer agent LLM
        llm_client = await self._get_llm_client()

        try:
            response = await llm_client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": self.config.system_prompt},
                    {"role": "user", "content": formatted_input},
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                stop=self.config.stop_words,
            )

            if response.choices and response.choices[0].message:
                content = response.choices[0].message.content or ""
                return self._extract_answer(content)
            return ""

        except Exception as e:
            logger.error(f"Answer agent LLM call failed: {e}")
            return f"Error: {str(e)}"

    async def answer_batch(
        self, queries: List[str], max_concurrent: int = 10
    ) -> List[str]:
        """
        Get answers for a batch of queries concurrently.

        Args:
            queries: List of queries to answer.
            max_concurrent: Maximum concurrent requests.

        Returns:
            List of answer texts.
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def answer_with_sem(query: str) -> str:
            async with semaphore:
                return await self.answer(query)

        return await asyncio.gather(*[answer_with_sem(q) for q in queries])

    async def close(self):
        """Close all client connections."""
        if self._llm_client is not None:
            await self._llm_client.close()
            self._llm_client = None
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
