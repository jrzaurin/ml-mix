# this is the code in this tutorial:
# https://docs.llamaindex.ai/en/stable/examples/low_level/oss_ingestion_retrieval/
# It has allowed me to navigate a bit the guts of the llama_index library but
# in general it is not very useful. Most of the 'pain' comes from having to
# set up postgres. The rest is just high-level API calls to the llama_index
# library.


from typing import Any, List, Optional

import psycopg2
from llama_index.core import QueryBundle
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.readers.file import PyMuPDFReader
from llama_index.vector_stores.postgres import PGVectorStore

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")

# model_url = "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML/resolve/main/llama-2-13b-chat.ggmlv3.q4_0.bin"
model_url = "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q4_0.gguf"


llm = LlamaCPP(
    # You can pass in the URL to a GGML model to downlssoad it automatically
    model_url=model_url,
    # optionally, you can set the path to a pre-downloaded model instead of model_url
    model_path=None,
    temperature=0.1,
    max_new_tokens=256,
    # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
    context_window=3900,
    # kwargs to pass to __call__()
    generate_kwargs={},
    # kwargs to pass to __init__()
    # set to at least 1 to use GPU
    model_kwargs={"n_gpu_layers": 1},
    verbose=True,
)


db_name = "vector_db"
host = "localhost"
password = "neutrino"
port = "5432"
user = "postgres"

# conn = psycopg2.connect(connection_string)
conn = psycopg2.connect(
    dbname="postgres",
    host=host,
    password=password,
    port=port,
    user=user,
)
conn.autocommit = True

with conn.cursor() as c:
    c.execute(f"DROP DATABASE IF EXISTS {db_name}")
    c.execute(f"CREATE DATABASE {db_name}")

vector_store = PGVectorStore.from_params(
    database=db_name,
    host=host,
    password=password,
    port=port,
    user=user,
    table_name="llama2_paper",
    embed_dim=384,  # openai embedding dimension
)


loader = PyMuPDFReader()
documents = loader.load(file_path="./data/llama2.pdf")

text_parser = SentenceSplitter(
    chunk_size=1024,
    # separator=" ",
)

text_chunks = []
# maintain relationship with source doc index, to help inject doc metadata in (3)
doc_idxs = []
for doc_idx, doc in enumerate(documents):
    cur_text_chunks = text_parser.split_text(doc.text)
    text_chunks.extend(cur_text_chunks)
    doc_idxs.extend([doc_idx] * len(cur_text_chunks))


nodes = []
for idx, text_chunk in enumerate(text_chunks):
    node = TextNode(
        text=text_chunk,
    )
    src_doc = documents[doc_idxs[idx]]
    node.metadata = src_doc.metadata
    nodes.append(node)


for node in nodes:
    node_embedding = embed_model.get_text_embedding(
        node.get_content(metadata_mode="all")  # type: ignore[arg-type]
    )
    node.embedding = node_embedding


vector_store.add(nodes)


query_str = "Can you tell me about the key concepts for safety finetuning"


query_embedding = embed_model.get_query_embedding(query_str)


query_mode = "default"
# query_mode = "sparse"
# query_mode = "hybrid"

vector_store_query = VectorStoreQuery(
    query_embedding=query_embedding, similarity_top_k=2, mode=query_mode  # type: ignore[arg-type]
)

query_result = vector_store.query(vector_store_query)
# print(query_result.nodes[0].get_content())


nodes_with_scores = []
for index, node in enumerate(query_result.nodes):
    score: Optional[float] = None
    if query_result.similarities is not None:
        score = query_result.similarities[index]
    nodes_with_scores.append(NodeWithScore(node=node, score=score))


class VectorDBRetriever(BaseRetriever):
    """Retriever over a postgres vector store."""

    def __init__(
        self,
        vector_store: PGVectorStore,
        embed_model: Any,
        query_mode: str = "default",
        similarity_top_k: int = 2,
    ) -> None:
        """Init params."""
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._query_mode = query_mode
        self._similarity_top_k = similarity_top_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve."""
        query_embedding = embed_model.get_query_embedding(query_bundle.query_str)
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self._similarity_top_k,
            mode=self._query_mode,  # type: ignore[arg-type]
        )
        query_result = vector_store.query(vector_store_query)

        nodes_with_scores = []
        for index, node in enumerate(query_result.nodes):
            score: Optional[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[index]
            nodes_with_scores.append(NodeWithScore(node=node, score=score))

        return nodes_with_scores


retriever = VectorDBRetriever(
    vector_store, embed_model, query_mode="default", similarity_top_k=2
)

query_engine = RetrieverQueryEngine.from_args(retriever, llm=llm)
query_str = "How does Llama 2 perform compared to other open-source models?"

response = query_engine.query(query_str)

print(str(response))

print(response.source_nodes[0].get_content())

print("done")
