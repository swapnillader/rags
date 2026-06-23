import os
import asyncio
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    Settings,
    StorageContext, 
    load_index_from_storage
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.tools import QueryEngineTool, FunctionTool

# 1. MODERN IMPORTS FOR LLAMAINDEX v0.13+ WORKFLOW AGENTS
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.workflow import Context

from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.postprocessor import LLMRerank

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.1)

Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

Settings.chunk_size = 512    # Cut pieces in half compared to the default 1024
Settings.chunk_overlap = 50   # Overlap pieces by ~35 words to protect context

PERSIST_DIR = "basics/storage"
LOAD_DIR = "basics/data"

async def main():
    reader = SimpleDirectoryReader(LOAD_DIR, filename_as_id=True)
    documents = reader.load_data()

    if not os.path.exists(PERSIST_DIR):
        print("Building a brand-new index...")
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        print("Loading existing index from disk...")
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
        refreshed_docs = index.refresh_ref_docs(documents)
        if any(refreshed_docs):
            index.storage_context.persist(persist_dir=PERSIST_DIR)

    vector_retriever = index.as_retriever(similarity_top_k=10)
    nodes = list(index.docstore.docs.values())
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=10)

    hybrid_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=10,
        num_queries=4,
        mode="reciprocal_rerank",
        use_async=False
    )
    reranker = LLMRerank(choice_batch_size=5, top_n=3)

    query_engine = RetrieverQueryEngine.from_args(
        retriever=hybrid_retriever,
        node_postprocessors=[reranker],
        llm=Settings.llm
    )

    document_tool = QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name="document_search_tool",
        description="Use this tool to find deep answers, summaries, facts, and specifics inside the loaded local documents."
    )

    def multiply(a: float, b: float) -> float:
        """Multiply two numbers together and return the exact float math calculation results."""
        return a * b

    calculator_tool = FunctionTool.from_defaults(fn=multiply)

    agent = ReActAgent(
        tools=[document_tool, calculator_tool],
        llm=Settings.llm,
        verbose=True,
    )
    
    ctx = Context(agent)

    print("\n--- Autonomous Workflow ReAct Agent Ready! ---")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        
        print("Bot is thinking...")
        response = await agent.run(user_msg=user_input, ctx=ctx)
        print(f"Bot: {response}\n")

if __name__ == "__main__":
    # Kick off the async runtime container
    asyncio.run(main())