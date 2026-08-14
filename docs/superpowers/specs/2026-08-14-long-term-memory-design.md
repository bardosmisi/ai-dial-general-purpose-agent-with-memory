# Long-Term Memory System Design

**Date:** 2026-08-14  
**Status:** Approved  
**Implementation Approach:** Bottom-Up (Storage First)

## Overview

Implement long-term memory capabilities for the DIAL General Purpose Agent, enabling it to:
- Proactively store user information across conversations
- Semantically search stored memories to personalize responses
- Delete all memories on user request

This is a simplified version focused on core functionality: storage in DIAL bucket, vector similarity search, and automatic deduplication.

## Architecture

### Component Layers

```
┌─────────────────────────────────────────┐
│   Agent + System Prompt (prompts.py)   │
│  (Decides when to store/search memory)  │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│          Tool Layer (3 tools)           │
│  StoreMemoryTool | SearchMemoryTool |   │
│           DeleteMemoryTool              │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│   Storage Layer (LongTermMemoryStore)   │
│  - DIAL bucket file persistence         │
│  - In-memory cache                      │
│  - Sentence embeddings                  │
│  - FAISS vector search                  │
│  - Deduplication (cosine similarity)    │
└─────────────────────────────────────────┘
```

### Data Flow

1. **Storage Flow:**
   - User shares information → Agent detects important fact → Calls `store_memory` tool → 
   - Tool extracts parameters → Storage layer encodes to vector → Saves to DIAL bucket → Updates cache

2. **Search Flow:**
   - User asks question → Agent decides context needed → Calls `search_memory` tool →
   - Storage checks if deduplication needed → Encodes query to vector → FAISS search →
   - Returns top-k memories → Tool formats as markdown → Agent uses in response

3. **Delete Flow:**
   - User requests deletion → Agent calls `delete_all_memories` tool →
   - Storage deletes file from bucket → Clears cache → Returns confirmation

### Storage Format

**File Location:** `files/{user_bucket}/__long-memories/data.json`

**JSON Structure:**
```json
{
  "memories": [
    {
      "data": {
        "id": 1723456789,
        "content": "User lives in Paris",
        "importance": 0.9,
        "category": "personal_info",
        "topics": ["location"]
      },
      "embedding": [0.123, -0.456, ..., 0.789]
    }
  ],
  "updated_at": "2026-08-14T10:30:00Z",
  "last_deduplicated_at": "2026-08-14T10:30:00Z"
}
```

**Cache Key:** File path (e.g., `files/bucket-uuid/__long-memories/data.json`)
- Unique per user
- Persists across conversations for same user
- Shared across all tool instances

## Storage Layer Implementation

### Class: `LongTermMemoryStore`

**Location:** `task/tools/memory/memory_store.py`

#### Initialization
```python
def __init__(self, endpoint: str):
    self.endpoint = endpoint
    self.model = SentenceTransformer('all-MiniLM-L6-v2')  # 384 dimensions
    self.cache: dict[str, MemoryCollection] = {}
    faiss.omp_set_num_threads(1)  # Debugger compatibility
```

#### Method: `_get_memory_file_path()`
- Get user's appdata bucket via `dial_client.get_app_home()`
- Return: `f"files/{bucket}/__long-memories/data.json"`

#### Method: `_load_memories(api_key: str) -> MemoryCollection`
1. Create `AsyncDial` client (api_version: `'2025-01-01-preview'`)
2. Get memory file path
3. Check cache by path key, return if exists
4. Try to download file:
   - Success: Decode UTF-8, parse JSON, create `MemoryCollection.model_validate(data)`
   - FileNotFoundError: Create new empty `MemoryCollection()` with current timestamp
5. Store in cache and return

#### Method: `_save_memories(api_key: str, memories: MemoryCollection)`
1. Create `AsyncDial` client
2. Get memory file path
3. Set `memories.updated_at = datetime.now(UTC)`
4. Serialize: `memories.model_dump_json()` (no indentation for smaller file size)
5. Upload to DIAL bucket via `dial_client.upload_file()`
6. Update cache

#### Method: `add_memory(api_key, content, importance, category, topics) -> str`
1. Load existing memories
2. Encode content: `self.model.encode([content])[0].tolist()`
3. Create memory:
   ```python
   memory = Memory(
       data=MemoryData(
           id=int(datetime.now(UTC).timestamp()),
           content=content,
           importance=importance,
           category=category,
           topics=topics
       ),
       embedding=embedding
   )
   ```
4. Append to collection: `memories.memories.append(memory)`
5. Save to bucket and cache
6. Return success message

#### Method: `search_memories(api_key, query, top_k=5) -> list[MemoryData]`
1. Load memories
2. Return empty list if no memories
3. Check `_needs_deduplication()`, if true: `await _deduplicate_and_save()`
4. Extract embeddings: `embeddings = np.array([m.embedding for m in memories.memories])`
5. Normalize embeddings: `faiss.normalize_L2(embeddings)`
6. Encode and normalize query: 
   ```python
   query_vec = self.model.encode([query])[0].reshape(1, -1)
   faiss.normalize_L2(query_vec)
   ```
7. Create FAISS index: `index = faiss.IndexFlatIP(embeddings.shape[1])`
8. Add embeddings: `index.add(embeddings)`
9. Search: `distances, indices = index.search(query_vec, min(top_k, len(memories.memories)))`
10. Return: `[memories.memories[i].data for i in indices[0]]`

#### Method: `_needs_deduplication(collection: MemoryCollection) -> bool`
```python
if len(collection.memories) <= 10:
    return False
if collection.last_deduplicated_at is None:
    return True
return (datetime.now(UTC) - collection.last_deduplicated_at) > timedelta(hours=24)
```

#### Method: `_deduplicate_and_save(api_key, collection) -> MemoryCollection`
1. Deduplicated = `_deduplicate_fast(collection.memories)`
2. Update collection: `collection.memories = deduplicated`
3. Set `collection.last_deduplicated_at = datetime.now(UTC)`
4. Save to bucket
5. Return updated collection

#### Method: `_deduplicate_fast(memories: list[Memory]) -> list[Memory]`
**Algorithm:** O(n log n) using FAISS batch neighbor search

1. If ≤1 memory, return as-is
2. Extract and normalize embeddings:
   ```python
   embeddings = np.array([m.embedding for m in memories])
   faiss.normalize_L2(embeddings)
   ```
3. Build FAISS index and search for k=11 neighbors (self + 10 nearest):
   ```python
   index = faiss.IndexFlatIP(embeddings.shape[1])
   index.add(embeddings)
   distances, indices = index.search(embeddings, min(11, len(memories)))
   ```
4. Mark duplicates:
   ```python
   to_remove = set()
   for i, (dists, neighs) in enumerate(zip(distances, indices)):
       if i in to_remove:
           continue
       for dist, neighbor_idx in zip(dists[1:], neighs[1:]):  # Skip self
           if dist > 0.75:  # Cosine similarity threshold
               # Keep higher importance
               if memories[i].data.importance >= memories[neighbor_idx].data.importance:
                   to_remove.add(neighbor_idx)
               else:
                   to_remove.add(i)
                   break
   ```
5. Return: `[m for i, m in enumerate(memories) if i not in to_remove]`

#### Method: `delete_all_memories(api_key: str) -> str`
1. Create `AsyncDial` client
2. Get memory file path
3. Delete file from bucket: `dial_client.delete_file(path)`
4. Remove from cache if exists
5. Return confirmation message

## Tool Layer Implementation

### StoreMemoryTool

**File:** `task/tools/memory/memory_store_tool.py`

```python
@property
def name(self) -> str:
    return "store_memory"

@property
def description(self) -> str:
    return """Store important information about the user for long-term memory.
    Use this proactively when the user shares personal details, preferences, goals, or context.
    Examples: location, workplace, name, preferences, plans, important facts.
    Store atomic facts (one fact per memory) with clear, concise language."""

@property
def parameters(self) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The memory content to store. Should be a clear, concise fact about the user."
            },
            "category": {
                "type": "string",
                "description": "Category of the info (e.g., 'preferences', 'personal_info', 'goals', 'plans', 'context')",
                "default": "general"
            },
            "importance": {
                "type": "number",
                "description": "Importance score between 0 and 1. Higher means more important to remember.",
                "minimum": 0,
                "maximum": 1,
                "default": 0.5
            },
            "topics": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Related topics or tags for the memory",
                "default": []
            }
        },
        "required": ["content", "category"]
    }

async def _execute(self, tool_call_params: ToolCallParams) -> str:
    args = json.loads(tool_call_params.tool_call.function.arguments)
    result = await self.memory_store.add_memory(
        api_key=tool_call_params.api_key,
        content=args["content"],
        importance=args.get("importance", 0.5),
        category=args.get("category", "general"),
        topics=args.get("topics", [])
    )
    tool_call_params.stage.append_content(result)
    return result
```

### SearchMemoryTool

**File:** `task/tools/memory/memory_search_tool.py`

```python
@property
def name(self) -> str:
    return "search_memory"

@property
def description(self) -> str:
    return """Search stored memories using semantic similarity.
    Use this when answering questions that could benefit from user context
    (location, preferences, personal details, past conversations).
    Returns relevant memories ranked by similarity."""

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
    results = await self.memory_store.search_memories(
        api_key=tool_call_params.api_key,
        query=args["query"],
        top_k=args.get("top_k", 5)
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
```

### DeleteMemoryTool

**File:** `task/tools/memory/memory_delete_tool.py`

```python
@property
def name(self) -> str:
    return "delete_all_memories"

@property
def description(self) -> str:
    return """Permanently delete all stored memories about the user.
    Use ONLY when the user explicitly requests to forget everything or wipe their memory.
    This action cannot be undone. Use with extreme caution."""

@property
def parameters(self) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {}
    }

async def _execute(self, tool_call_params: ToolCallParams) -> str:
    result = await self.memory_store.delete_all_memories(
        api_key=tool_call_params.api_key
    )
    tool_call_params.stage.append_content(result)
    return result
```

## System Prompt Design

**File:** `task/prompts.py`

**Strategy:** Make memory operations part of the agent's identity and natural behavior.

**Key Sections:**

1. **Role & Capabilities:**
   - Establish agent as having long-term memory
   - Explain memory persists across conversations

2. **Proactive Storage Rules:**
   - When to store: user shares name, location, workplace, preferences, goals, plans, context
   - How to store: atomic facts, clear language, appropriate importance scores
   - Importance guidelines: 0.8-0.9 for critical info (name, location), 0.5-0.7 for preferences
   - Store silently without announcing to user

3. **Proactive Retrieval Rules:**
   - Before answering personalized questions, search memories first
   - Examples: weather questions → search location, recommendations → search preferences
   - Use retrieved context naturally in responses

4. **Memory Quality:**
   - One fact per memory
   - Use present tense ("User lives in Paris", not "User said they live in Paris")
   - Choose appropriate categories and topics

**Prompt Template Structure:**
```
You are a general-purpose AI assistant with long-term memory capabilities.

# MEMORY SYSTEM

You have access to a persistent memory system that stores information about the user across all conversations.

## When to Store Memories

PROACTIVELY store information when users mention:
- Personal details: name, age, location, occupation, family
- Preferences: favorite foods, music, programming languages, workflows
- Goals and plans: learning objectives, projects, travel plans
- Important context: pets, hobbies, recurring problems

Store memories SILENTLY - don't announce "I'll remember that" unless the user explicitly asks you to remember something.

## How to Store Memories

Use the store_memory tool with:
- content: Clear, concise fact in present tense
- category: 'personal_info', 'preferences', 'goals', 'plans', 'context', or 'general'
- importance: 0.8-0.9 for critical facts (name, location), 0.5-0.7 for preferences, 0.3-0.5 for context
- topics: Relevant keywords for search

## When to Search Memories

BEFORE answering questions that benefit from user context, search memories:
- Location-specific questions (weather, recommendations, time zones)
- Personalized advice (tech stack, workflows, preferences)
- Remembering past conversations or context

Use retrieved memories naturally without saying "I found in my memory..."

# YOUR OTHER TOOLS

[List of other tools: web search, code interpreter, image generation, RAG, file extraction]

# RESPONSE GUIDELINES

- Be helpful, accurate, and concise
- Use memory operations seamlessly in conversation
- Respect user privacy - memories are private and persistent
```

## Integration in app.py

**Location:** `task/app.py`, method `_create_tools()` line 64

Add after RAG and Python interpreter tools:

```python
StoreMemoryTool(memory_store=self.memory_store),
SearchMemoryTool(memory_store=self.memory_store),
DeleteMemoryTool(memory_store=self.memory_store),
```

## Error Handling

### Storage Layer
- **File not found:** Return empty `MemoryCollection` with initialized timestamps
- **DIAL errors:** Propagate to tool layer for error messaging
- **Empty search:** Return empty list (tool formats as "No memories found")
- **Single memory deduplication:** Check collection size before FAISS operations

### Tool Layer
- **Invalid parameters:** Handled by LLM's schema validation
- **Storage exceptions:** Caught by `BaseTool.execute()`, returns error message to LLM
- **Empty results:** User-friendly messages

### Deduplication
- **Identical embeddings:** FAISS handles, keeps highest importance
- **All duplicates:** Keep at least one (highest importance)
- **Importance ties:** Deterministic ordering (first in list wins)

## Testing Strategy

### Phase 1: Storage Layer Unit Testing
- Test `add_memory()` with mock DIAL client
- Test `search_memories()` with sample embeddings
- Test `_deduplicate_fast()` with known duplicate sets
- Verify FAISS similarity threshold (0.75)

### Phase 2: Integration Testing
1. **Store 5-10 memories:**
   - "My name is John"
   - "I live in Paris"
   - "I work at Google"
   - "I prefer Python over JavaScript"
   - "I'm learning Spanish"

2. **Test search in new conversation:**
   - Ask "What's the weather?" → Should retrieve Paris location → Search web for Paris weather
   - Ask "Help me with code" → Should retrieve Python preference → Provide Python examples

3. **Test deletion:**
   - Request "Delete all my memories"
   - Verify `data.json` removed from DIAL bucket
   - Verify cache cleared
   - Verify new conversation has no context

## Success Criteria

- ✅ Memories persist across conversations in DIAL bucket
- ✅ Semantic search returns relevant memories (not just keyword matching)
- ✅ Deduplication runs automatically and removes >75% similar memories
- ✅ Agent proactively stores user info without being asked
- ✅ Agent searches memories before answering contextual questions
- ✅ Delete command wipes all memory data
- ✅ No performance degradation up to 1000 memories

## Implementation Order

1. Implement `LongTermMemoryStore` class (all methods)
2. Implement `StoreMemoryTool`
3. Implement `SearchMemoryTool`
4. Implement `DeleteMemoryTool`
5. Add tools to `app.py`
6. Write system prompt in `prompts.py`
7. Test with docker-compose environment
8. Iterate on system prompt based on LLM behavior

## Non-Goals (Out of Scope)

- Distributed cache across multiple instances
- Memory expiration or TTL
- User-level access control (DIAL handles auth)
- Memory editing or updating (only add/delete)
- Per-conversation memory isolation
- Memory export/import
- Advanced deduplication strategies beyond cosine similarity
