import os

from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.text_splitter import MarkdownTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader,
)
from langchain_openai import OpenAIEmbeddings

from .FileMetadata import ContentMetadata
from .utils import pprint


def setup_embedder(cache_path):
    print("-" * 80)
    print(f"Setting up embedding cache: {cache_path}")

    underlying_embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
    )

    local_store = LocalFileStore(cache_path)
    os.makedirs(local_store.root_path, exist_ok=True)
    print("n_cache_emb:", len(os.listdir(local_store.root_path)))

    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings,
        local_store,
        namespace=underlying_embeddings.model,
    )
    print("n_cache_emb:", len(os.listdir(local_store.root_path)))
    return cached_embedder, local_store


def load_docs(docs_path, loader_kwargs={}, splitter_kwargs={}):
    # ---------------------
    # Loader kwargs defaults
    loader_kwargs.setdefault("autodetect_encoding", True)
    loader_kwargs.setdefault("glob", "**/*.md")
    loader_kwargs.setdefault("show_progress", True)
    loader_kwargs.setdefault("recursive", True)
    # loader_kwargs.setdefault("use_multithreading", True)
    # loader_kwargs.setdefault("max_concurrency", 4)
    # loader_kwargs.setdefault("loader_cls", UnstructuredMarkdownLoader)

    # loader_kwargs = dict(
    #     autodetect_encoding=True,
    # )
    # loader_kwargs.setdefault("loader_kwargs", text_loader_kwargs)

    print("Loader kwargs:")
    pprint(loader_kwargs)

    # ---------------------
    # Splitter kwargs defaults
    splitter_kwargs.setdefault("chunk_size", 2048)
    splitter_kwargs.setdefault("chunk_overlap", 512)

    print("Splitter kwargs:")
    pprint(splitter_kwargs)

    # ---------------------

    print("=" * 80)
    print(f"Loading documents: {docs_path}")

    loader = DirectoryLoader(
        docs_path,
        **loader_kwargs,
    )
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    text_splitter = MarkdownTextSplitter(**splitter_kwargs)

    # ---------------------

    docs = loader.load()

    print(f"Loaded {len(docs)} documents")

    for doc in docs:
        print("-" * 80)
        d = vars(doc)
        print(d.keys())
        print(doc.metadata)
        source = doc.metadata.get("source", None)
        meta = ContentMetadata.from_path(source)
        doc.metadata.update(meta.json())
        print(doc.metadata)

    print("-" * 80)
    print("Splitting documents...")
    # Split
    splits = text_splitter.split_documents(docs)
    print("num_splits:", len(splits))

    for split in splits:
        split.metadata["splitter"] = text_splitter.__class__.__name__

    return splits
