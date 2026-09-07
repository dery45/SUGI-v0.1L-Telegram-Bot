import time
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError


def ping_with_retry(mongo: MongoClient, label: str, attempts: int = 3, base_wait: float = 5.0) -> None:
    """A7: ping MongoDB dengan retry + backoff linear (5s, 10s).
    See docs/decisions.md#a7
    """
    for attempt in range(attempts):
        try:
            mongo.admin.command("ping")
            return
        except ServerSelectionTimeoutError as e:
            if attempt == attempts - 1:
                raise
            wait = base_wait * (attempt + 1)
            print(f"  {label}: Mongo ping failed (attempt {attempt+1}/{attempts}), "
                  f"retrying in {wait}s: {e}")
            time.sleep(wait)


def build_insight_llm(model_name: str, temperature: float, timeout: int = 240):
    """A8: OllamaLLM konsisten (timeout 240s dipakai semua service insight).
    P2-2: reads OLLAMA_HOST_INSIGHT to isolate background generation from user path.
    G2: keep_alive shortened to 60s for insight instance to free VRAM/RAM between
        infrequent cycles (3600s-86400s), trades cold reload for headroom.
    See docs/decisions.md#a8, #p2-2, #g2
    """
    import os
    from langchain_ollama.llms import OllamaLLM
    return OllamaLLM(
        model=model_name,
        temperature=temperature,
        repeat_penalty=1.1,
        num_ctx=4096,
        keep_alive=60,  # G2: 60s vs default 5m, unloads between insight cycles
        base_url=os.getenv("OLLAMA_HOST_INSIGHT", "http://127.0.0.1:11434"),
        client_kwargs={"timeout": timeout},
    )


def get_rag_context(query: str, k: int, label: str, content_chars: int) -> str:
    try:
        from services.vectorCSV import vector_store as _vs
        docs = _vs.similarity_search(query, k=k)
        if docs:
            return label + "\n" + "\n".join(d.page_content[:content_chars] for d in docs)
    except Exception:
        pass
    return ""


def get_weather_context(k: int, label: str, content_chars: int) -> str:
    try:
        from services.vectorWeather import weather_store as _ws
        docs = _ws.similarity_search("cuaca pertanian Indonesia", k=k)
        if docs:
            return label + "\n" + "\n".join(d.page_content[:content_chars] for d in docs)
    except Exception:
        pass
    return ""


def get_plant_context(commodity: str, chroma_host: str, chroma_port: int, embed_model: str) -> str:
    try:
        import chromadb as _cdb
        from langchain_chroma import Chroma
        from langchain_ollama import OllamaEmbeddings
        _client = _cdb.HttpClient(host=chroma_host, port=chroma_port)
        _emb = OllamaEmbeddings(model=embed_model)
        _pc = Chroma(collection_name="plant_data", client=_client, embedding_function=_emb)
        docs = _pc.similarity_search(commodity, k=1)
        if docs:
            return f"INFO TANAMAN:\n{docs[0].page_content[:300]}"
    except Exception:
        pass
    return ""