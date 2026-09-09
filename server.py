import os

from dotenv import load_dotenv
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
import functools

from fastmcp import FastMCP
from fastmcp.tools import Tool
from fastmcp.server.event_store import EventStore
import resources
import tools

# Désactive le parallélisme interne du Tokenizer HuggingFace pour éviter les deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

project_dir = Path(__file__).resolve().parent
env_path = project_dir / ".env"
load_dotenv(dotenv_path=env_path)

# ====================================================================
# 1. CRÉATION DU POOL DE THREADS (LE "SWEET SPOT" FINAL)
# max_workers=2 : On autorise 2 recherches SIMULTANÉES.
# Sachant que PyTorch est limité à 12 threads par le script de recherche,
# cela fera 24 cœurs actifs sur vos 30. C'est l'équilibre parfait entre
# un temps d'attente nul pour les utilisateurs et un processeur qui respire !
# ====================================================================
ai_thread_pool = ThreadPoolExecutor(max_workers=2)

mcp = FastMCP(
    name="ServeurHarmonIA",
)


# ====================================================================
# 2. CRÉATION DE LA FONCTION WRAPPER ASYNCHRONE
# ====================================================================
async def async_retrieve_search_documents(search_terms: str, limit: int = 20, tags: list = None) -> dict:
    loop = asyncio.get_running_loop()

    func = functools.partial(
        tools.retrieve_search_documents,
        search_terms=search_terms,
        tags=tags,
        limit=limit
    )

    # Exécution dans le thread pool
    results = await loop.run_in_executor(ai_thread_pool, func)

    # Formatage JSON sécurisé
    return {"result": results}


async def async_retrieve_document_context(document_id: str, query: str, top_k: int = 3, window_size: int = 1) -> dict:
    """Wrapper asynchrone pour l'extraction de contexte RAG."""
    loop = asyncio.get_running_loop()

    func = functools.partial(
        tools.retrieve_document_context,
        document_id=document_id,
        query=query,
        top_k=top_k,
        window_size=window_size
    )

    # Exécution dans le thread pool pour ne pas bloquer l'Event Loop
    results = await loop.run_in_executor(ai_thread_pool, func)

    # Formatage JSON sécurisé
    return {"result": results}


# --- ENREGISTREMENT DES OUTILS ---
# Search tools are optional when qdrant_client is unavailable. Register them
# only if the underlying functions were successfully imported.
if tools.retrieve_search_documents is not None:
    mcp.add_tool(
        Tool.from_function(
            async_retrieve_search_documents,
            name="retrieve_search_documents",
        )
    )

if tools.retrieve_document_context is not None:
    mcp.add_tool(
        Tool.from_function(
            async_retrieve_document_context,
            name="retrieve_document_context",
        )
    )

if tools.get_available_tags is not None:
    mcp.add_tool(
        Tool.from_function(
            tools.get_available_tags,
            name="get_available_tags",
        )
    )

if tools.get_document_file is not None:
    mcp.add_tool(
        Tool.from_function(
            tools.get_document_file,
            name="get_document_file",
        )
    )

mcp.add_tool(
    Tool.from_function(
        tools.upload_model,
        name="upload_model",
    )
)

mcp.add_tool(
    Tool.from_function(
        tools.get_model,
        name="get_model",
    )
)

mcp.add_tool(
    Tool.from_function(
        tools.touch_model,
        name="touch_model",
    )
)

mcp.add_tool(
    Tool.from_function(
        tools.list_models,
        name="list_models",
    )
)

mcp.add_tool(
    Tool.from_function(
        tools.rename_model,
        name="rename_model",
    )
)

mcp.add_tool(
    Tool.from_function(
        tools.delete_model,
        name="delete_model",
    )
)

mcp.add_tool(
    Tool.from_function(
        tools.add_class,
        name="add_class",
    )
)

mcp.add_tool(
    Tool.from_function(
        tools.add_attribute,
        name="add_attribute",
    )
)

mcp.add_tool(
    Tool.from_function(
        tools.add_connector,
        name="add_connector",
    )
)

if tools.retrieve_documents is not None:
    mcp.add_tool(
        Tool.from_function(
            tools.retrieve_documents,
            name="retrieve_documents",
        )
    )

if tools.get_style_guide is not None:
    mcp.add_tool(
        Tool.from_function(
            tools.get_style_guide,
            name="get_style_guide",
        )
    )

if tools.plan_workflow_with_tools is not None:
    mcp.add_tool(
        Tool.from_function(
            tools.plan_workflow_with_tools,
            name="plan_workflow_with_tools",
        )
    )

if tools.metadata_checker is not None:
    mcp.add_tool(
        Tool.from_function(
            tools.metadata_checker,
            name="metadata_checker",
        )
    )

if tools.reuse_check is not None:
    mcp.add_tool(
        Tool.from_function(
            tools.reuse_check,
            name="reuse_check",
        )
    )

if tools.validator_check is not None:
    mcp.add_tool(
        Tool.from_function(
            tools.validator_check,
            name="validator_check",
        )
    )

if tools.style_guide_check is not None:
    mcp.add_tool(
        Tool.from_function(
            tools.style_guide_check,
            name="style_guide_check",
        )
    )


# --- ENREGISTREMENT DES RESSOURCES MCP ---
@mcp.resource("resource://model/{user}/{session_name}")
async def get_model_resource(user: str, session_name: str) -> str:
    import json
    data = resources.get_model(user, session_name)
    return json.dumps(data, ensure_ascii=False, indent=2)


@mcp.resource("resource://Style_Guide")
async def get_style_guide_resource() -> str:
    return await resources.get_style_guide()


event_store = EventStore()
app = mcp.http_app(
    event_store=event_store,
    retry_interval=2000,
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
