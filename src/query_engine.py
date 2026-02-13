"""
Query engine — handles RAG retrieval and LLM response generation.

Takes a user question, retrieves relevant code chunks from the vector store,
assembles a prompt with context, and generates a response via Ollama.
"""

import logging

from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core import PromptTemplate
from llama_index.llms.ollama import Ollama

from src.config import OLLAMA_HOST, OLLAMA_MODEL, TOP_K

logger = logging.getLogger(__name__)

# System prompt for the code documentation assistant
CODE_DOC_PROMPT = PromptTemplate(
    "You are a code documentation assistant. You help developers understand "
    "codebases by answering questions about how the code works, where "
    "functionality is implemented, API endpoints, dependencies, and architecture.\n\n"
    "Use the following code context to answer the question. When referencing code, "
    "mention the file path so the developer can find it. If the context doesn't "
    "contain enough information to answer fully, say so and suggest what the "
    "developer might look for.\n\n"
    "--- Code Context ---\n"
    "{context_str}\n"
    "--- End Context ---\n\n"
    "Question: {query_str}\n\n"
    "Answer:"
)


def create_query_engine(index: VectorStoreIndex) -> RetrieverQueryEngine:
    """
    Create a query engine from an existing index.

    Configures:
    - Retriever: top-k similarity search from ChromaDB
    - Postprocessor: filters out low-similarity results
    - LLM: Ollama with the configured model tier
    - Prompt: Code documentation-specific prompt template
    """
    llm = Ollama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_HOST,
        request_timeout=120.0,
    )

    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=TOP_K,
    )

    # Filter out chunks with very low similarity scores
    postprocessor = SimilarityPostprocessor(similarity_cutoff=0.3)

    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        node_postprocessors=[postprocessor],
        llm=llm,
        text_qa_template=CODE_DOC_PROMPT,
    )

    logger.info(
        f"Query engine ready (model={OLLAMA_MODEL}, top_k={TOP_K})"
    )
    return query_engine


def query(engine: RetrieverQueryEngine, question: str) -> dict:
    """
    Run a query and return the response with source information.

    Returns:
        dict with 'answer', 'sources' (list of file paths), and 'source_nodes'
    """
    logger.info(f"Query: {question[:100]}...")

    response = engine.query(question)

    # Extract source file paths from the response metadata
    sources = []
    for node in response.source_nodes:
        file_path = node.metadata.get("file_path", "unknown")
        score = node.score if node.score is not None else 0.0
        sources.append({
            "file_path": file_path,
            "score": round(score, 3),
            "language": node.metadata.get("language", "unknown"),
            "text_preview": node.text[:200] + "..." if len(node.text) > 200 else node.text,
        })

    logger.info(f"Response generated from {len(sources)} source chunks")

    return {
        "answer": str(response),
        "sources": sources,
        "source_nodes": response.source_nodes,
    }
