# The MIT License (MIT)
# Copyright © 2024 OpenKaito

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
import time
import typing
from datetime import datetime

import bittensor as bt
from dotenv import load_dotenv
from elasticsearch import Elasticsearch

import openkaito
from openkaito.base.miner import BaseMinerNeuron
from openkaito.crawlers.twitter.apidojo import ApiDojoTwitterCrawler
from openkaito.protocol import (
    DiscordSearchSynapse,
    SearchSynapse,
    SemanticSearchSynapse,
    StructuredSearchSynapse,
    TextEmbeddingSynapse,
)
from openkaito.search.ranking import HeuristicRankingModel
from openkaito.search.structured_search_engine import StructuredSearchEngine
from openkaito.utils.embeddings import openai_embeddings_tensor
from openkaito.utils.version import compare_version, get_version

import openai
from sentence_transformers import SentenceTransformer
from transformers.utils import logging
import torch.nn.functional as F
logging.set_verbosity_error()

import google.generativeai as genai
import asyncio
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

class Miner(BaseMinerNeuron):
    """
    Your miner neuron class. You should use this class to define your miner's behavior. In particular, you should replace the forward function with your own logic. You may also want to override the blacklist and priority functions according to your needs.

    This class inherits from the BaseMinerNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a miner such as blacklisting unrecognized hotkeys, prioritizing requests based on stake, and forwarding requests to the forward function. If you need to define custom
    """

    def __init__(self):
        super(Miner, self).__init__()

        load_dotenv()

        search_client = Elasticsearch(
            os.environ["ELASTICSEARCH_HOST"],
            basic_auth=(
                os.environ["ELASTICSEARCH_USERNAME"],
                os.environ["ELASTICSEARCH_PASSWORD"],
            ),
            verify_certs=False,
            ssl_show_warn=False,
        )

        # for ranking recalled results
        ranking_model = HeuristicRankingModel(length_weight=0.8, age_weight=0.2)

        # optional, for crawling data
        twitter_crawler = (
            # MicroworldsTwitterCrawler(os.environ["APIFY_API_KEY"])
            ApiDojoTwitterCrawler(os.environ["APIFY_API_KEY"])
            if os.environ.get("APIFY_API_KEY")
            else None
        )

        self.structured_search_engine = StructuredSearchEngine(
            search_client=search_client,
            relevance_ranking_model=ranking_model,
            twitter_crawler=twitter_crawler,
            recall_size=self.config.neuron.search_recall_size,
        )

        # openai embeddings model
        self.client = openai.OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            organization=os.getenv("OPENAI_ORGANIZATION"),
            project=os.getenv("OPENAI_PROJECT"),
            max_retries=3,
        )

        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        # self.model_type = "google"
        self.model_type = "sentence-transformers"
        # self.model = SentenceTransformer("dunzhang/stella_en_400M_v5", trust_remote_code=True).cuda()
        # self.model = SentenceTransformer("nvidia/NV-Embed-v2", trust_remote_code=True)
        self.model = SentenceTransformer("output/finetuned_model", trust_remote_code=True).cuda()
        self.device = "cuda"

    async def forward_search(self, query: SearchSynapse) -> SearchSynapse:
        """
        Processes the incoming Search synapse by performing a search operation on the crawled data.

        Args:
            query (SearchSynapse): The synapse object containing the query information.

        Returns:
            SearchSynapse: The synapse object with the 'results' field set to list of the 'Document'.
        """
        start_time = datetime.now()
        bt.logging.info(f"received SearchSynapse: ", query)
        self.check_version(query)

        if not self.config.neuron.disable_crawling:
            crawl_size = max(self.config.neuron.crawl_size, query.size)
            self.structured_search_engine.crawl_and_index_data(
                query_string=query.query_string,
                author_usernames=None,
                # crawl and index more data than needed to ensure we have enough to rank
                max_size=crawl_size,
            )

        ranked_docs = self.structured_search_engine.search(query)

        bt.logging.debug(f"{len(ranked_docs)} ranked_docs", ranked_docs)
        query.results = ranked_docs
        end_time = datetime.now()
        elapsed_time = (end_time - start_time).total_seconds()
        bt.logging.info(
            f"processed SearchSynapse in {elapsed_time} seconds",
        )
        return query

    async def forward_structured_search(
        self, query: StructuredSearchSynapse
    ) -> StructuredSearchSynapse:

        start_time = datetime.now()
        bt.logging.info(
            f"received StructuredSearchSynapse... timeout:{query.timeout}s ", query
        )
        self.check_version(query)

        # miners may adjust this timeout config by themselves according to their own crawler speed and latency
        if query.timeout > 12:
            # do crawling and indexing, otherwise search from the existing index directly
            crawl_size = max(self.config.neuron.crawl_size, query.size)
            self.structured_search_engine.crawl_and_index_data(
                query_string=query.query_string,
                author_usernames=query.author_usernames,
                # crawl and index more data than needed to ensure we have enough to rank
                max_size=crawl_size,
            )

        # disable crawling for structured search by default
        ranked_docs = self.structured_search_engine.search(query)
        bt.logging.debug(f"{len(ranked_docs)} ranked_docs", ranked_docs)
        query.results = ranked_docs
        end_time = datetime.now()
        elapsed_time = (end_time - start_time).total_seconds()
        bt.logging.info(
            f"processed StructuredSearchSynapse in {elapsed_time} seconds",
        )
        return query

    async def forward_semantic_search(
        self, query: SemanticSearchSynapse
    ) -> SemanticSearchSynapse:

        start_time = datetime.now()
        bt.logging.info(
            f"received SemanticSearchSynapse... timeout:{query.timeout}s ", query
        )
        self.check_version(query)

        ranked_docs = self.structured_search_engine.vector_search(query)
        bt.logging.debug(f"{len(ranked_docs)} ranked_docs", ranked_docs)
        query.results = ranked_docs
        end_time = datetime.now()
        elapsed_time = (end_time - start_time).total_seconds()
        bt.logging.info(
            f"processed SemanticSearchSynapse in {elapsed_time} seconds",
        )
        return query

    async def forward_discord_search(
        self, query: DiscordSearchSynapse
    ) -> DiscordSearchSynapse:

        start_time = datetime.now()
        bt.logging.info(
            f"received DiscordSearchSynapse... timeout:{query.timeout}s ", query
        )
        self.check_version(query)

        ranked_docs = self.structured_search_engine.discord_search(query)
        bt.logging.debug(f"{len(ranked_docs)} ranked_docs", ranked_docs)
        query.results = ranked_docs
        end_time = datetime.now()
        elapsed_time = (end_time - start_time).total_seconds()
        bt.logging.info(
            f"processed DiscordSearchSynapse in {elapsed_time} seconds",
        )
        return query

    async def process_text(self,text):
        if self.model_type == "sentence-transformers":
            return self.model.encode(text, convert_to_tensor=True, normalize_embeddings=True, device=self.device,
                                     show_progress_bar=True, batch_size=64, num_workers=8)
        elif self.model_type == "openai":
            pass
        elif self.model_type == "google":
            return genai.embed_content(
                model="models/text-embedding-004",
                task_type="retrieval_query",
                content=text,
                output_dimensionality=512,
            )["embedding"]


    # Example of a text embedding function
    async def forward_text_embedding(
        self, query: TextEmbeddingSynapse
    ) -> TextEmbeddingSynapse:
        texts = query.texts
        dimensions = query.dimensions

        time_start = time.time()
        texts = [text.replace("\n", " ") for text in texts]
        bt.logging.info(
            f"received TextEmbedding Synapse... timeout:{query.timeout}s with text lens: {str(len(query.texts))} "
            f"and dimensions: {query.dimensions}",
        )

        if self.model_type == "sentence-transformers":
            # Sentence Transformers
            # embeddings = self.model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
            tasks = [self.process_text(text) for text in texts]
            embeddings = await asyncio.gather(*tasks)
            query.results = [embedding[:query.dimensions].tolist() for embedding in embeddings]

            time_end = time.time()
            elapsed_time = time_end - time_start
            bt.logging.info(
                f"processed TextEmbedding Synapse in {elapsed_time} seconds, with embeddings shape: {len(query.results[0])}",
            )
        elif self.model_type == "openai":
            # OpenAI embeddings
            embeddings = openai_embeddings_tensor(
               self.client, texts, dimensions=dimensions, model="text-embedding-3-large"
            )
            query.results = embeddings.tolist()

            time_end = time.time()
            elapsed_time = time_end - time_start

            bt.logging.info(
                f"processed TextEmbedding Synapse in {elapsed_time} seconds, with embeddings shape: {len(query.results)}",
            )
        elif self.model_type == "google":
            # Google embeddings
            tasks = [self.process_text(text) for text in texts]
            embeddings = await asyncio.gather(*tasks)
            query.results = [embedding for embedding in embeddings]

            time_end = time.time()
            elapsed_time = time_end - time_start

            bt.logging.info(
                f"processed TextEmbedding Synapse in {elapsed_time} seconds, with embeddings shape: {len(query.results)}",
            )
        return query

    def print_info(self):
        metagraph = self.metagraph
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)

        log = (
            "Miner | "
            f"Epoch:{self.step} | "
            f"UID:{self.uid} | "
            f"Block:{self.block} | "
            f"Stake:{metagraph.S[self.uid]} | "
            f"Rank:{metagraph.R[self.uid]} | "
            f"Trust:{metagraph.T[self.uid]} | "
            f"Consensus:{metagraph.C[self.uid] } | "
            f"Incentive:{metagraph.I[self.uid]} | "
            f"Emission:{metagraph.E[self.uid]}"
        )
        bt.logging.info(log)

    def check_version(self, query):
        """
        Check the version of the incoming request and log a warning if it is newer than the miner's running version.
        """
        if (
            query.version is not None
            and compare_version(query.version, get_version()) > 0
        ):
            bt.logging.warning(
                f"Received request with version {query.version}, is newer than miner running version {get_version()}. You may updating the repo and restart the miner."
            )


# This is the main function, which runs the miner.
if __name__ == "__main__":
    with Miner() as miner:
        while True:
            miner.print_info()
            time.sleep(30)
