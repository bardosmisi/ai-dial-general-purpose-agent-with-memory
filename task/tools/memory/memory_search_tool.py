import json
from typing import Any

from task.tools.base import BaseTool
from task.tools.memory._models import MemoryData
from task.tools.memory.memory_store import LongTermMemoryStore
from task.tools.models import ToolCallParams


class SearchMemoryTool(BaseTool):
    """
    Tool for searching long-term memories about the user.

    Performs semantic search over stored memories to find relevant information.
    """

    def __init__(self, memory_store: LongTermMemoryStore):
        self.memory_store = memory_store


    @property
    def name(self) -> str:
        return "search_memory"

    @property
    def description(self) -> str:
        return """Search stored memories using semantic similarity. Use this when answering questions that could benefit from user context (location, preferences, personal details, past conversations). Returns relevant memories ranked by similarity."""

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query. Can be a question or keywords to find relevant memories."
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of most relevant memories to return.",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 5
                }
            },
            "required": ["query"]
        }


    async def _execute(self, tool_call_params: ToolCallParams) -> str:
        args = json.loads(tool_call_params.tool_call.function.arguments)

        query = args["query"]
        top_k = args.get("top_k", 5)

        results = await self.memory_store.search_memories(
            api_key=tool_call_params.api_key,
            query=query,
            top_k=top_k
        )

        if not results:
            result_text = "No memories found."
        else:
            result_text = "## Found Memories:\n\n"
            for i, memory in enumerate(results, 1):
                result_text += f"**{i}. {memory.category}**\n"
                result_text += f"- Content: {memory.content}\n"
                if memory.topics:
                    result_text += f"- Topics: {', '.join(memory.topics)}\n"
                result_text += "\n"

        tool_call_params.stage.append_content(result_text)
        return result_text
