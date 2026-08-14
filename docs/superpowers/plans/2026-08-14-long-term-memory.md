# Long-Term Memory System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement long-term memory capabilities for DIAL agent with storage, semantic search, and deduplication.

**Architecture:** Bottom-up approach - storage layer first (LongTermMemoryStore with FAISS), then tool wrappers (Store/Search/Delete), then integration with app.py, finally system prompt to enable proactive memory usage.

**Tech Stack:** Python, FAISS (vector search), SentenceTransformers (embeddings), DIAL SDK (bucket storage), Pydantic (models)

## Global Constraints

- Python 3.9+ compatibility
- Use `SentenceTransformer('all-MiniLM-L6-v2')` for embeddings (384 dimensions)
- FAISS cosine similarity threshold: 0.75 for deduplication
- Memory file location: `files/{bucket}/__long-memories/data.json`
- Deduplication triggers: >10 memories AND >24 hours since last dedup
- API version for AsyncDial: `'2025-01-01-preview'`
- Set `faiss.omp_set_num_threads(1)` for debugger compatibility

---

## File Structure

**New files:**
- None (all files already exist with TODOs)

**Modified files:**
- `task/tools/memory/memory_store.py` - Core storage logic with FAISS
- `task/tools/memory/memory_store_tool.py` - Store memory tool
- `task/tools/memory/memory_search_tool.py` - Search memory tool
- `task/tools/memory/memory_delete_tool.py` - Delete memory tool
- `task/app.py` - Add memory tools to application
- `task/prompts.py` - System prompt for proactive memory usage

**Existing models (no changes needed):**
- `task/tools/memory/_models.py` - Pydantic models already defined

---

### Task 1: Storage Layer - Initialization and File Path

**Files:**
- Modify: `task/tools/memory/memory_store.py:14-42`

**Interfaces:**
- Consumes: None
- Produces: 
  - `LongTermMemoryStore.__init__(endpoint: str)` - Initializes storage with embedding model and cache
  - `LongTermMemoryStore._get_memory_file_path(dial_client: AsyncDial) -> str` - Returns file path for user's memories

- [ ] **Step 1: Implement `__init__` method**

Replace the `__init__` method (lines 27-32):

```python
def __init__(self, endpoint: str):
    self.endpoint = endpoint
    self.model = SentenceTransformer('all-MiniLM-L6-v2')
    self.cache: dict[str, MemoryCollection] = {}
    faiss.omp_set_num_threads(1)
```

- [ ] **Step 2: Implement `_get_memory_file_path` method**

Replace the method (lines 34-41):

```python
async def _get_memory_file_path(self, dial_client: AsyncDial) -> str:
    """Get the path to the memory file in DIAL bucket."""
    app_home = await dial_client.get_app_home()
    return f"files/{app_home}/__long-memories/data.json"
```

- [ ] **Step 3: Test initialization manually**

Run Python interpreter:
```bash
python -c "from task.tools.memory.memory_store import LongTermMemoryStore; store = LongTermMemoryStore('http://localhost:8080'); print('Init success:', store.endpoint, type(store.model), type(store.cache))"
```

Expected: `Init success: http://localhost:8080 <class 'sentence_transformers.SentenceTransformer.SentenceTransformer'> <class 'dict'>`

- [ ] **Step 4: Commit initialization**

```bash
git add task/tools/memory/memory_store.py
git commit -m "feat: implement LongTermMemoryStore initialization and file path method"
```

---

### Task 2: Storage Layer - Load and Save

**Files:**
- Modify: `task/tools/memory/memory_store.py:43-76`

**Interfaces:**
- Consumes:
  - `LongTermMemoryStore._get_memory_file_path(dial_client: AsyncDial) -> str`
- Produces:
  - `LongTermMemoryStore._load_memories(api_key: str) -> MemoryCollection` - Loads memories from bucket or cache
  - `LongTermMemoryStore._save_memories(api_key: str, memories: MemoryCollection)` - Saves memories to bucket and cache

- [ ] **Step 1: Implement `_load_memories` method**

Replace the method (lines 43-62):

```python
async def _load_memories(self, api_key: str) -> MemoryCollection:
    """Load memories from cache or DIAL bucket."""
    dial_client = AsyncDial(
        base_url=self.endpoint,
        api_key=api_key,
        api_version='2025-01-01-preview'
    )
    
    memory_file_path = await self._get_memory_file_path(dial_client)
    
    # Check cache first
    if memory_file_path in self.cache:
        return self.cache[memory_file_path]
    
    # Try to load from bucket
    try:
        response = await dial_client.files.download(memory_file_path)
        content = response.decode('utf-8')
        data = json.loads(content)
        collection = MemoryCollection.model_validate(data)
    except Exception:
        # File doesn't exist, create new collection
        collection = MemoryCollection()
    
    self.cache[memory_file_path] = collection
    return collection
```

- [ ] **Step 2: Implement `_save_memories` method**

Replace the method (lines 64-76):

```python
async def _save_memories(self, api_key: str, memories: MemoryCollection):
    """Save memories to DIAL bucket and update cache."""
    dial_client = AsyncDial(
        base_url=self.endpoint,
        api_key=api_key,
        api_version='2025-01-01-preview'
    )
    
    memory_file_path = await self._get_memory_file_path(dial_client)
    
    memories.updated_at = datetime.now(UTC)
    
    json_content = memories.model_dump_json()
    
    await dial_client.files.upload(
        path=memory_file_path,
        content=json_content.encode('utf-8')
    )
    
    self.cache[memory_file_path] = memories
```

- [ ] **Step 3: Commit load and save methods**

```bash
git add task/tools/memory/memory_store.py
git commit -m "feat: implement memory load and save with DIAL bucket integration"
```

---

### Task 3: Storage Layer - Add Memory

**Files:**
- Modify: `task/tools/memory/memory_store.py:78-91`

**Interfaces:**
- Consumes:
  - `LongTermMemoryStore._load_memories(api_key: str) -> MemoryCollection`
  - `LongTermMemoryStore._save_memories(api_key: str, memories: MemoryCollection)`
- Produces:
  - `LongTermMemoryStore.add_memory(api_key: str, content: str, importance: float, category: str, topics: list[str]) -> str`

- [ ] **Step 1: Implement `add_memory` method**

Replace the method (lines 78-91):

```python
async def add_memory(self, api_key: str, content: str, importance: float, category: str, topics: list[str]) -> str:
    """Add a new memory to storage."""
    memories = await self._load_memories(api_key)
    
    # Encode content to embedding
    embedding = self.model.encode([content])[0].tolist()
    
    # Create memory with timestamp ID
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
    
    memories.memories.append(memory)
    
    await self._save_memories(api_key, memories)
    
    return f"Memory stored successfully: '{content}'"
```

- [ ] **Step 2: Commit add memory**

```bash
git add task/tools/memory/memory_store.py
git commit -m "feat: implement add_memory with embedding generation"
```

---

### Task 4: Storage Layer - Deduplication Helper Methods

**Files:**
- Modify: `task/tools/memory/memory_store.py:109-126`

**Interfaces:**
- Consumes: None
- Produces:
  - `LongTermMemoryStore._needs_deduplication(collection: MemoryCollection) -> bool`
  - `LongTermMemoryStore._deduplicate_and_save(api_key: str, collection: MemoryCollection) -> MemoryCollection`

- [ ] **Step 1: Implement `_needs_deduplication` method**

Replace the method (lines 109-114):

```python
def _needs_deduplication(self, collection: MemoryCollection) -> bool:
    """Check if deduplication is needed (>24 hours since last deduplication)."""
    if len(collection.memories) <= 10:
        return False
    if collection.last_deduplicated_at is None:
        return True
    time_since_dedup = datetime.now(UTC) - collection.last_deduplicated_at
    return time_since_dedup > timedelta(hours=24)
```

- [ ] **Step 2: Implement `_deduplicate_and_save` method**

Replace the method (lines 116-126):

```python
async def _deduplicate_and_save(self, api_key: str, collection: MemoryCollection) -> MemoryCollection:
    """
    Deduplicate memories synchronously and save the result.
    Returns the updated collection.
    """
    deduplicated = self._deduplicate_fast(collection.memories)
    collection.memories = deduplicated
    collection.last_deduplicated_at = datetime.now(UTC)
    await self._save_memories(api_key, collection)
    return collection
```

- [ ] **Step 3: Commit deduplication helpers**

```bash
git add task/tools/memory/memory_store.py
git commit -m "feat: implement deduplication helper methods"
```

---

### Task 5: Storage Layer - FAISS Deduplication Algorithm

**Files:**
- Modify: `task/tools/memory/memory_store.py:128-143`

**Interfaces:**
- Consumes: None
- Produces:
  - `LongTermMemoryStore._deduplicate_fast(memories: list[Memory]) -> list[Memory]`

- [ ] **Step 1: Implement `_deduplicate_fast` method**

Replace the method (lines 128-143):

```python
def _deduplicate_fast(self, memories: list[Memory]) -> list[Memory]:
    """
    Fast deduplication using FAISS batch search with cosine similarity.

    Strategy:
    - Find k nearest neighbors for each memory using cosine similarity
    - Mark duplicates based on similarity threshold (cosine similarity > 0.75)
    - Keep memory with higher importance
    """
    if len(memories) <= 1:
        return memories
    
    # Extract embeddings as numpy array
    embeddings = np.array([m.embedding for m in memories], dtype='float32')
    
    # Normalize for cosine similarity (use inner product with normalized vectors)
    faiss.normalize_L2(embeddings)
    
    # Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    # Search for k=11 neighbors (self + 10 nearest)
    k = min(11, len(memories))
    distances, indices = index.search(embeddings, k)
    
    # Mark duplicates
    to_remove = set()
    for i in range(len(memories)):
        if i in to_remove:
            continue
        
        # Skip first neighbor (self with distance 1.0)
        for j in range(1, k):
            neighbor_idx = indices[i][j]
            similarity = distances[i][j]
            
            if similarity > 0.75:  # Cosine similarity threshold
                # Decide which to keep based on importance
                if memories[i].data.importance >= memories[neighbor_idx].data.importance:
                    to_remove.add(neighbor_idx)
                else:
                    to_remove.add(i)
                    break  # This memory is marked for removal, stop checking
    
    # Return deduplicated list
    return [m for idx, m in enumerate(memories) if idx not in to_remove]
```

- [ ] **Step 2: Commit FAISS deduplication**

```bash
git add task/tools/memory/memory_store.py
git commit -m "feat: implement FAISS-based deduplication algorithm"
```

---

### Task 6: Storage Layer - Search and Delete

**Files:**
- Modify: `task/tools/memory/memory_store.py:93-107,145-157`

**Interfaces:**
- Consumes:
  - `LongTermMemoryStore._load_memories(api_key: str) -> MemoryCollection`
  - `LongTermMemoryStore._needs_deduplication(collection: MemoryCollection) -> bool`
  - `LongTermMemoryStore._deduplicate_and_save(api_key: str, collection: MemoryCollection) -> MemoryCollection`
  - `LongTermMemoryStore._get_memory_file_path(dial_client: AsyncDial) -> str`
- Produces:
  - `LongTermMemoryStore.search_memories(api_key: str, query: str, top_k: int) -> list[MemoryData]`
  - `LongTermMemoryStore.delete_all_memories(api_key: str) -> str`

- [ ] **Step 1: Implement `search_memories` method**

Replace the method (lines 93-107):

```python
async def search_memories(self, api_key: str, query: str, top_k: int = 5) -> list[MemoryData]:
    """
    Search memories using semantic similarity.

    Returns:
        List of MemoryData objects (without embeddings)
    """
    collection = await self._load_memories(api_key)
    
    if not collection.memories:
        return []
    
    # Check and perform deduplication if needed
    if self._needs_deduplication(collection):
        collection = await self._deduplicate_and_save(api_key, collection)
    
    # Extract embeddings
    embeddings = np.array([m.embedding for m in collection.memories], dtype='float32')
    
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Encode and normalize query
    query_embedding = self.model.encode([query])[0].reshape(1, -1).astype('float32')
    faiss.normalize_L2(query_embedding)
    
    # Build FAISS index and search
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    k = min(top_k, len(collection.memories))
    distances, indices = index.search(query_embedding, k)
    
    # Return MemoryData objects (without embeddings)
    return [collection.memories[idx].data for idx in indices[0]]
```

- [ ] **Step 2: Implement `delete_all_memories` method**

Replace the method (lines 145-157):

```python
async def delete_all_memories(self, api_key: str) -> str:
    """
    Delete all memories for the user.

    Removes the memory file from DIAL bucket and clears the cache
    for the current conversation.
    """
    dial_client = AsyncDial(
        base_url=self.endpoint,
        api_key=api_key,
        api_version='2025-01-01-preview'
    )
    
    memory_file_path = await self._get_memory_file_path(dial_client)
    
    await dial_client.files.delete(memory_file_path)
    
    # Clear from cache
    if memory_file_path in self.cache:
        del self.cache[memory_file_path]
    
    return "All memories have been permanently deleted."
```

- [ ] **Step 3: Commit search and delete**

```bash
git add task/tools/memory/memory_store.py
git commit -m "feat: implement search_memories with FAISS and delete_all_memories"
```

---

### Task 7: Tool Layer - StoreMemoryTool

**Files:**
- Modify: `task/tools/memory/memory_store_tool.py:24-54`

**Interfaces:**
- Consumes:
  - `LongTermMemoryStore.add_memory(api_key: str, content: str, importance: float, category: str, topics: list[str]) -> str`
- Produces:
  - `StoreMemoryTool` with complete implementation

- [ ] **Step 1: Implement `name` property**

Replace lines 25-27:

```python
@property
def name(self) -> str:
    return "store_memory"
```

- [ ] **Step 2: Implement `description` property**

Replace lines 29-33:

```python
@property
def description(self) -> str:
    return """Store important information about the user for long-term memory. Use this proactively when the user shares personal details, preferences, goals, or context. Examples: location, workplace, name, preferences, plans, important facts. Store atomic facts (one fact per memory) with clear, concise language."""
```

- [ ] **Step 3: Implement `parameters` property**

Replace lines 35-42:

```python
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
```

- [ ] **Step 4: Implement `_execute` method**

Replace lines 44-54:

```python
async def _execute(self, tool_call_params: ToolCallParams) -> str:
    args = json.loads(tool_call_params.tool_call.function.arguments)
    
    content = args["content"]
    category = args.get("category", "general")
    importance = args.get("importance", 0.5)
    topics = args.get("topics", [])
    
    result = await self.memory_store.add_memory(
        api_key=tool_call_params.api_key,
        content=content,
        importance=importance,
        category=category,
        topics=topics
    )
    
    tool_call_params.stage.append_content(result)
    return result
```

- [ ] **Step 5: Commit StoreMemoryTool**

```bash
git add task/tools/memory/memory_store_tool.py
git commit -m "feat: implement StoreMemoryTool"
```

---

### Task 8: Tool Layer - SearchMemoryTool

**Files:**
- Modify: `task/tools/memory/memory_search_tool.py:21-50`

**Interfaces:**
- Consumes:
  - `LongTermMemoryStore.search_memories(api_key: str, query: str, top_k: int) -> list[MemoryData]`
- Produces:
  - `SearchMemoryTool` with complete implementation

- [ ] **Step 1: Implement `name` property**

Replace lines 22-24:

```python
@property
def name(self) -> str:
    return "search_memory"
```

- [ ] **Step 2: Implement `description` property**

Replace lines 26-30:

```python
@property
def description(self) -> str:
    return """Search stored memories using semantic similarity. Use this when answering questions that could benefit from user context (location, preferences, personal details, past conversations). Returns relevant memories ranked by similarity."""
```

- [ ] **Step 3: Implement `parameters` property**

Replace lines 32-37:

```python
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
```

- [ ] **Step 4: Implement `_execute` method**

Replace lines 40-50:

```python
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
```

- [ ] **Step 5: Commit SearchMemoryTool**

```bash
git add task/tools/memory/memory_search_tool.py
git commit -m "feat: implement SearchMemoryTool"
```

---

### Task 9: Tool Layer - DeleteMemoryTool

**Files:**
- Modify: `task/tools/memory/memory_delete_tool.py:20-40`

**Interfaces:**
- Consumes:
  - `LongTermMemoryStore.delete_all_memories(api_key: str) -> str`
- Produces:
  - `DeleteMemoryTool` with complete implementation

- [ ] **Step 1: Implement `name` property**

Replace lines 20-22:

```python
@property
def name(self) -> str:
    return "delete_all_memories"
```

- [ ] **Step 2: Implement `description` property**

Replace lines 24-28:

```python
@property
def description(self) -> str:
    return """Permanently delete all stored memories about the user. Use ONLY when the user explicitly requests to forget everything or wipe their memory. This action cannot be undone. Use with extreme caution."""
```

- [ ] **Step 3: Implement `parameters` property**

Replace lines 30-32:

```python
@property
def parameters(self) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {}
    }
```

- [ ] **Step 4: Implement `_execute` method**

Replace lines 35-40:

```python
async def _execute(self, tool_call_params: ToolCallParams) -> str:
    result = await self.memory_store.delete_all_memories(
        api_key=tool_call_params.api_key
    )
    
    tool_call_params.stage.append_content(result)
    return result
```

- [ ] **Step 5: Commit DeleteMemoryTool**

```bash
git add task/tools/memory/memory_delete_tool.py
git commit -m "feat: implement DeleteMemoryTool"
```

---

### Task 10: Integration - Add Tools to Application

**Files:**
- Modify: `task/app.py:63-65`

**Interfaces:**
- Consumes:
  - `StoreMemoryTool(memory_store: LongTermMemoryStore)`
  - `SearchMemoryTool(memory_store: LongTermMemoryStore)`
  - `DeleteMemoryTool(memory_store: LongTermMemoryStore)`
  - `self.memory_store` from `GeneralPurposeAgentApplication`
- Produces:
  - Complete tool list with memory tools

- [ ] **Step 1: Add memory tools to `_create_tools` method**

Replace the TODO comment at line 63-65 with:

```python
            StoreMemoryTool(memory_store=self.memory_store),
            SearchMemoryTool(memory_store=self.memory_store),
            DeleteMemoryTool(memory_store=self.memory_store),
```

The complete tools list should now be:

```python
async def _create_tools(self) -> list[BaseTool]:
    tools: list[BaseTool] = [
        ImageGenerationTool(endpoint=DIAL_ENDPOINT),
        FileContentExtractionTool(endpoint=DIAL_ENDPOINT),
        RagTool(
            endpoint=DIAL_ENDPOINT,
            deployment_name=DEPLOYMENT_NAME,
            document_cache=DocumentCache.create()
        ),
        await PythonCodeInterpreterTool.create(
            mcp_url="http://localhost:8050/mcp",
            tool_name="execute_code",
            dial_endpoint=DIAL_ENDPOINT
        ),
        
        StoreMemoryTool(memory_store=self.memory_store),
        SearchMemoryTool(memory_store=self.memory_store),
        DeleteMemoryTool(memory_store=self.memory_store),
    ]

    tools.extend(await self._get_mcp_tools("http://localhost:8051/mcp"))

    return tools
```

- [ ] **Step 2: Commit integration**

```bash
git add task/app.py
git commit -m "feat: integrate memory tools into agent application"
```

---

### Task 11: System Prompt Design

**Files:**
- Modify: `task/prompts.py:1-8`

**Interfaces:**
- Consumes:
  - `store_memory` tool (proactive storage)
  - `search_memory` tool (semantic search)
  - `delete_all_memories` tool (explicit deletion)
- Produces:
  - `SYSTEM_PROMPT` string that instructs agent to use memory proactively

- [ ] **Step 1: Write system prompt**

Replace entire file content:

```python
SYSTEM_PROMPT = """You are a general-purpose AI assistant with long-term memory capabilities.

# MEMORY SYSTEM

You have access to a persistent memory system that stores information about the user across all conversations. This memory persists even if the user starts a new conversation.

## When to Store Memories

PROACTIVELY store information when users mention:
- Personal details: name, age, location, occupation, family members
- Preferences: favorite foods, music, programming languages, tools, workflows
- Goals and plans: learning objectives, projects, career goals, travel plans
- Important context: pets, hobbies, recurring problems, dislikes

Store memories SILENTLY during natural conversation - don't announce "I'll remember that" unless the user explicitly asks you to remember something.

## How to Store Memories

Use the store_memory tool with these guidelines:
- content: Clear, concise fact in present tense (e.g., "User lives in Paris", not "User said they live in Paris")
- category: Choose from 'personal_info', 'preferences', 'goals', 'plans', 'context', or 'general'
- importance: 
  - 0.8-0.9 for critical facts (name, location, occupation)
  - 0.5-0.7 for preferences and useful context
  - 0.3-0.5 for minor details
- topics: Add relevant keywords to help with search (e.g., ["location", "france"] for Paris)

Store ONE fact per memory. If the user shares multiple facts, store each separately.

## When to Search Memories

BEFORE answering questions that could benefit from user context, search memories first:
- Location-specific questions: "What's the weather?" → search for location first
- Recommendations: "What should I learn next?" → search for goals, preferences
- Personalized advice: "Help me with code" → search for programming language preferences
- Context recall: "Remember what we discussed?" → search for relevant past topics

Use retrieved memories naturally in your responses. Don't say "I found in my memory..." - just use the context as if you naturally know it.

## Memory Quality

- Be specific: "User prefers Python over JavaScript" not "User likes Python"
- Use present tense: "User works at Google" not "User said they work at Google"
- Atomic facts: One piece of information per memory
- Avoid duplicates: Don't store the same fact multiple times

# YOUR OTHER TOOLS

You have access to additional tools for various tasks:
- Web search (DuckDuckGo) - for current information and web content
- Python code interpreter - for executing Python code and data analysis
- Image generation (DALL-E) - for creating images
- RAG search - for searching through uploaded documents
- File content extraction - for reading PDFs, CSVs, text files

# RESPONSE GUIDELINES

- Be helpful, accurate, and concise
- Use memory operations seamlessly - they should be invisible to the user
- Respect user privacy - memories are private and persistent
- When in doubt about storing something, store it - the user can always delete memories later
- Use natural language in responses, don't be robotic about memory operations
"""
```

- [ ] **Step 2: Commit system prompt**

```bash
git add task/prompts.py
git commit -m "feat: implement system prompt for proactive memory usage"
```

---

### Task 12: End-to-End Testing Setup

**Files:**
- None (manual testing with docker-compose)

**Interfaces:**
- Consumes: Complete implementation
- Produces: Verified working system

- [ ] **Step 1: Start docker-compose environment**

```bash
docker-compose up -d
```

Expected: All services start (chat, core, redis, python-interpreter, ddg-search)

- [ ] **Step 2: Start the agent application**

```bash
python -m task.app
```

Expected: Agent starts on port 5030, connects to DIAL Core

- [ ] **Step 3: Open DIAL Chat UI**

Navigate to: http://localhost:3000

Expected: Chat UI loads, shows "General Purpose Agent" in applications list

- [ ] **Step 4: Test memory storage (Conversation 1)**

In chat, send these messages one by one:
1. "My name is John"
2. "I live in Paris"
3. "I work at Google as a software engineer"
4. "I prefer Python over JavaScript"
5. "I'm currently learning Spanish"

Expected: Agent responds naturally to each message. Check agent logs for `store_memory` tool calls.

- [ ] **Step 5: Verify storage in DIAL bucket**

In Chat UI, check attachments panel for `__long-memories/data.json` file.

Expected: File exists with 5 memory entries

- [ ] **Step 6: Test memory search (Conversation 2 - New conversation)**

Start a NEW conversation, send:
"What's the weather like where I live?"

Expected: Agent calls `search_memory` tool first (check logs), retrieves Paris location, then searches web for Paris weather

- [ ] **Step 7: Test contextual memory retrieval**

In same conversation, send:
"Help me write some code to parse a JSON file"

Expected: Agent searches memory, finds Python preference, provides Python example (not JavaScript)

- [ ] **Step 8: Test memory deletion**

Send: "Delete all my memories"

Expected: Agent calls `delete_all_memories` tool, confirms deletion. Check attachments panel - `data.json` should be gone.

- [ ] **Step 9: Verify clean slate**

Start a NEW conversation, send:
"What's the weather?"

Expected: Agent asks for location (doesn't remember Paris anymore)

- [ ] **Step 10: Document test results**

Create file `docs/superpowers/testing/2026-08-14-memory-test-results.md`:

```markdown
# Long-Term Memory Testing Results

**Date:** 2026-08-14

## Test Cases

### ✅ Memory Storage
- Stored 5 memories across different categories
- Verified in DIAL bucket attachments
- Agent stored memories silently during conversation

### ✅ Memory Search
- New conversation retrieved location context
- Agent used search results naturally in response
- Code assistance used programming language preference

### ✅ Memory Deletion
- Delete command wiped all memories
- Verified file removed from bucket
- New conversation had no context

## Issues Found
[List any issues encountered]

## Notes
[Any observations about agent behavior, prompt effectiveness, etc.]
```

- [ ] **Step 11: Commit test documentation**

```bash
git add docs/superpowers/testing/2026-08-14-memory-test-results.md
git commit -m "docs: add end-to-end testing results for memory system"
```

---

## Implementation Complete

After completing all tasks, the long-term memory system will be fully functional with:
- ✅ Storage layer with FAISS vector search and deduplication
- ✅ Three tools for store/search/delete operations
- ✅ Integration with DIAL agent application
- ✅ System prompt for proactive memory usage
- ✅ End-to-end testing verification

The agent will proactively store user information, semantically search memories to personalize responses, and allow users to delete all memories on demand.
