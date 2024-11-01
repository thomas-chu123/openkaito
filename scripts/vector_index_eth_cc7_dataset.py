"""Vector Search of Eth Denver dataset using Elasticsearch and Transformers

This script extracts the Eth Denver dataset (open-sourced by https://portal.kaito.ai/events/ETHDenver2024 ), indexes the documents in Elasticsearch, and indexes the embeddings of the documents in Elasticsearch.
It also provides a test query to retrieve the top-k similar documents to the query.

This script is intentionally kept transparent and hackable, and miners may do their own customizations.
"""

import os
from pathlib import Path
import json
from tqdm import tqdm
import torch

from elasticsearch import Elasticsearch, helpers
from dotenv import load_dotenv

import concurrent.futures

from openkaito.utils.embeddings import pad_tensor, text_embedding, MAX_EMBEDDING_DIM


root_dir = __file__.split("scripts")[0]
index_name = "eth_cc7"


### Extract Eth Denver dataset
def extract_eth_cc7_dataset():
    """Extract Eth Denver dataset to datasets/eth_denver_dataset directory"""

    if os.path.exists(root_dir + "datasets/eth_cc7_dataset"):
        print(
            "Eth cc7 data already extracted to: ",
            root_dir + "datasets/eth_cc7_dataset",
        )
    else:
        import tarfile

        with tarfile.open(
            root_dir + "datasets/eth_cc7_dataset.tar.gz", "r:gz"
        ) as tar:
            tar.extractall(root_dir + "datasets")

        print(
            "Eth cc7 data extracted to: ", root_dir + "datasets/eth_cc7_dataset"
        )

    dataset_dir = root_dir + "datasets/eth_cc7_dataset"
    dataset_path = Path(dataset_dir)
    print(f"{len(list(dataset_path.glob('*.json')))} files in {dataset_dir}")


def init_eth_cc7_index(search_client):
    """Initialize Eth Denver index in Elasticsearch"""

    if not search_client.indices.exists(index=index_name):
        print("creating index...", index_name)
        search_client.indices.create(
            index=index_name,
            body={
                "mappings": {
                    "properties": {
                        "doc_id": {"type": "keyword"},
                        "episode_id": {"type": "keyword"},
                        "segment_id": {"type": "long"},
                        "episode_title": {"type": "text"},
                        "episode_url": {"type": "text"},
                        "created_at": {"type": "date"},
                        "company_name": {"type": "keyword"},
                        "segment_start_time": {"type": "float"},
                        "segment_end_time": {"type": "float"},
                        "text": {"type": "text"},
                        "speaker": {"type": "keyword"},
                        "embedding": {
                            "type": "dense_vector",
                            "dims": MAX_EMBEDDING_DIM,
                        },
                    }
                }
            },
        )
        print("Index created: ", index_name)
    else:
        print("Index already exists: ", index_name)


def drop_index(search_client, index_name):
    """Drop index in Elasticsearch"""

    if search_client.indices.exists(index=index_name):
        search_client.indices.delete(index=index_name)
        print("Index deleted: ", index_name)
    else:
        print("Index does not exist: ", index_name)


def indexing_docs(search_client):
    """Index documents in Elasticsearch"""

    dataset_dir = root_dir + "datasets/eth_cc7_dataset"
    dataset_path = Path(dataset_dir)

    num_files = len(list(dataset_path.glob("*.json")))
    print(f"Indexing {num_files} files in {dataset_dir}")
    for doc_file in tqdm(
        dataset_path.glob("*.json"), total=num_files, desc="Indexing docs"
    ):
        with open(doc_file, "r") as f:
            doc = json.load(f)
            search_client.index(index=index_name, body=doc, id=doc["doc_id"])

def new_indexing_embeddings(search_client, max_workers=20):
    """Index embeddings of documents in Elasticsearch with parallel processing"""

    try:
        def process_doc(doc):
            doc_id = doc["_id"]
            text = doc["_source"]["text"]
            embedding = text_embedding(text)[0]
            embedding = pad_tensor(embedding, max_len=MAX_EMBEDDING_DIM)
            search_client.update(
                index=index_name,
                id=doc_id,
                body={"doc": {"embedding": embedding.tolist()}, "doc_as_upsert": True},
                timeout="5m",
            )

        docs = helpers.scan(search_client, index=index_name)
        total_docs = search_client.count(index=index_name)["count"]

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(tqdm(executor.map(process_doc, docs), total=total_docs, desc="Indexing embeddings"))
    except Exception as e:
        print(e)

def indexing_embeddings(search_client):
    """Index embeddings of documents in Elasticsearch"""

    for doc in tqdm(
        helpers.scan(search_client, index=index_name),
        desc="Indexing embeddings",
        total=search_client.count(index=index_name)["count"],
    ):
        doc_id = doc["_id"]
        text = doc["_source"]["text"]
        embedding = text_embedding(text)[0]
        embedding = pad_tensor(embedding, max_len=MAX_EMBEDDING_DIM)
        search_client.update(
            index=index_name,
            id=doc_id,
            body={"doc": {"embedding": embedding.tolist()}, "doc_as_upsert": True},
        )


def test_retrieval(search_client, query, topk=5):
    """Test retrieval of top-k similar documents to query"""

    embedding = text_embedding(query)[0]
    embedding = pad_tensor(embedding, max_len=MAX_EMBEDDING_DIM)
    body = {
        "knn": {
            "field": "embedding",
            "query_vector": embedding.tolist(),
            "k": topk,
            "num_candidates": 5 * topk,
        },
        # "_source": {
        #     "excludes": ["embedding"],
        # },
    }

    response = search_client.search(index=index_name, body=body)
    return response


if __name__ == "__main__":
    load_dotenv()

    dataset_dir = root_dir + "datasets/eth_cc7_dataset"
    dataset_path = Path(dataset_dir)

    num_files = len(list(dataset_path.glob("*.json")))

    extract_eth_cc7_dataset()

    search_client = Elasticsearch(
        os.environ["ELASTICSEARCH_HOST"],
        basic_auth=(
            os.environ["ELASTICSEARCH_USERNAME"],
            os.environ["ELASTICSEARCH_PASSWORD"],
        ),
        verify_certs=False,
        ssl_show_warn=False,
        request_timeout=120,
    )

    # drop_index(search_client, index_name)
    init_eth_cc7_index(search_client)

    r = search_client.count(index=index_name)
    if r["count"] != num_files:
        print(
            f"Number of docs in {index_name}: {r['count']} != total files {num_files}, reindexing docs..."
        )
        indexing_docs(search_client)
    else:
        print(
            f"Number of docs in {index_name}: {r['count']} == total files {num_files}, no need to reindex docs"
        )

    # indexing_embeddings(search_client)
    new_indexing_embeddings(search_client)

    query = "What is the future of blockchain?"
    response = test_retrieval(search_client, query, topk=5)
    # print(response)
    for response in response["hits"]["hits"]:
        print(response["_source"]["created_at"])
        print(response["_source"]["episode_title"])
        print(response["_source"]["speaker"])
        print(response["_source"]["text"])
        print(response["_score"])
