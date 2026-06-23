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
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.postprocessor import LLMRerank

# NEW IMPORTS FOR PHASE 6 EVALUATION
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

# Standard Base Configurations
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.0) # Set temp to 0.0 for consistent testing!
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# Define the folder name where we want to save our data
PERSIST_DIR = "basics/storage"
LOAD_DIR = "basics/data"

async def main():
    # 1. Load the existing storage engine index from disk
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
    
    # 2. Build our standard retrieval pipeline structure
    vector_retriever = index.as_retriever(similarity_top_k=5)
    nodes = list(index.docstore.docs.values())
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=5)
    
    hybrid_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=5,
        num_queries=1,
        mode="reciprocal_rerank"
    )
    
    query_engine = RetrieverQueryEngine.from_args(
        retriever=hybrid_retriever,
        llm=Settings.llm
    )
    
    # 3. Initialize our Automated LLM Evaluator Judges
    # We use a separate evaluator instance to grade the output independently
    faithfulness_evaluator = FaithfulnessEvaluator(llm=Settings.llm)
    relevancy_evaluator = RelevancyEvaluator(llm=Settings.llm)
    
    # 4. Run a live test question against your system
    test_query = "Give me the list EDP product's"
    print(f"Sending test query: '{test_query}'...")
    
    response = query_engine.query(test_query)
    
    # 5. Execute the evaluation algorithms
    print("\n--- Running Evaluation Metrics ---")
    
    # Check for hallucinations
    faith_result = faithfulness_evaluator.evaluate_response(response=response)
    # Check for question assignment alignment
    rel_result = relevancy_evaluator.evaluate_response(query=test_query, response=response)
    
    # 6. Print the Scorecard
    print("\n================ EVALUATION SCORECARD ================")
    print(f"Response Given: {response}\n")
    print(f"FAITHFULNESS (No Hallucinations): {'✅ PASS' if faith_result.passing else '❌ FAIL'}")
    print(f"Feedback: {faith_result.feedback}")
    print(f"Score: {faith_result.score}\n")
    
    print(f"RELEVANCY (Answered the Prompt):  {'✅ PASS' if rel_result.passing else '❌ FAIL'}")
    print(f"Feedback: {rel_result.feedback}")
    print(f"Score: {rel_result.score}")
    print("======================================================")

if __name__ == "__main__":
    asyncio.run(main())