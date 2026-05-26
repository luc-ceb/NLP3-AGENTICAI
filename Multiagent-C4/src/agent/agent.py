
from typing import AsyncGenerator
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import  create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.store.memory import InMemoryStore
from langmem.short_term import SummarizationNode

from langchain_ollama import ChatOllama
from langgraph_supervisor import create_supervisor

from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain_community.tools.arxiv.tool import ArxivQueryRun

from .prompt import *
from src.tools import tools  
from .state import State

# ---------------------------------------------------------------------
# Configuración de modelos (all-Ollama)
# ---------------------------------------------------------------------
REASONING_MODEL = "qwen3:14b"
WORKER_MODEL    = "qwen3:8b"        # agentes simples + summarization
TEMPERATURE     = 0.1

def make_reasoning_llm():
    """LLM con thinking activo. El content queda limpio y el reasoning
    va a additional_kwargs['reasoning_content']."""
    return ChatOllama(
        model=REASONING_MODEL,
        temperature=TEMPERATURE,
        reasoning=True,
    )

def make_worker_llm():
    """LLM sin thinking. Doble candado: reasoning=False + think=False
    en model_kwargs por el bug conocido con qwen3 en langchain-ollama."""
    return ChatOllama(
        model=WORKER_MODEL,
        temperature=TEMPERATURE,
        reasoning=False,
        model_kwargs={"think": False},
    )

# ---------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------
arxiv = ArxivAPIWrapper(
    top_k_results = 3,
    ARXIV_MAX_QUERY_LENGTH = 300,
    load_max_docs = 3,
    load_all_available_meta = False,
    doc_content_chars_max = 10000
)

arxiv_tool = ArxivQueryRun(api_wrapper=arxiv)
tools.append(arxiv_tool)

checkpointer = InMemorySaver()  # For thread-level state persistence
memory_store = InMemoryStore() 

# ---------------------------------------------------------------------
# Agentes workers (sin thinking)
# ---------------------------------------------------------------------
summarizer_agent = create_react_agent(
    model=make_worker_llm(),
    prompt=summarizer_prompt,
    tools=[],
    name="SummarizerAgent",
    state_schema=State,
)

research_agent = create_react_agent(
    model=make_worker_llm(),
    prompt=system_prompt,
    tools=tools,
    name="ResearchAssistant",
    state_schema=State,
)

# ---------------------------------------------------------------------
# Summarization node (compresión de historial) — modelo simple
# ---------------------------------------------------------------------
summarization_node = SummarizationNode(
    token_counter=count_tokens_approximately,
    model=make_worker_llm(),
    max_tokens=768,
    max_summary_tokens=512,
    output_messages_key="llm_input_messages",
)

# ---------------------------------------------------------------------
# Supervisor (reasoning model)
# ---------------------------------------------------------------------
supervisor = create_supervisor(
    agents=[research_agent, summarizer_agent],
    model=make_reasoning_llm(),
    prompt=supervisor_prompt,
    pre_model_hook=summarization_node,
).compile(checkpointer=checkpointer, store=memory_store)

# ---------------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------------
async def query(q: str, session_id: str) -> AsyncGenerator[str, None]:
    state = {"messages": [HumanMessage(content=q)]}
    config = {"configurable": {"thread_id": session_id}}
    async for event in supervisor.astream_events(state, version="v2", config=config):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            chunk = event["data"].get("chunk")
            if chunk and hasattr(chunk, "content"):
                text = chunk.content
                if text:
                    yield text

async def interactive(session_id: str = None):
    print(f"Research Assistant (LangGraph + Ollama)")
    print(f"  Supervisor: {REASONING_MODEL} (reasoning ON)")
    print(f"  Workers:    {WORKER_MODEL} (reasoning OFF)")
    print("Type 'exit' to quit.\n")
    while True:
        try:
            q = input("\nQuery> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye"); break
        if q.lower() in ("exit", "quit"):
            break
        async for response in query(q, session_id):
            print(response, end="", flush=True)