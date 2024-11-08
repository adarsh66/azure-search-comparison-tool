from datetime import datetime
import os
import json
import openai
import logging
from azure.storage.blob import BlobServiceClient
import requests


class CreateIndex:
    def __init__(
        self,
        file_path,
        azure_credentials,
        openai_apikey,
        openai_endpoint,
        openai_deploymentname,
        azure_storage_account,
        azure_storage_container,
    ):
        self.file_path = file_path
        self.azure_credentials = azure_credentials
        self.openai_apikey = openai_apikey
        self.openai_endpoint = openai_endpoint
        self.openai_deploymentname = openai_deploymentname
        self.azure_storage_account = azure_storage_account
        self.azure_storage_container = azure_storage_container
        self.openai_api_version = "2023-05-15"
        self.openai_api_type = "azure_ad"

        if not self._is_valid_json_file():
            raise ValueError("Invalid JSON file or file does not exist.")

    def _is_valid_json_file(self):
        if not os.path.isfile(self.file_path):
            return False

        try:
            with open(self.file_path) as file:
                data = json.load(file)
                if not all(key in data for key in ["id", "title", "content"]):
                    return False
                return True
        except (ValueError, OSError):
            return False

    def upload_files_to_blob(self, save_path=""):
        blob_service = BlobServiceClient(
            account_url=f"https://{self.azure_storage_account}.blob.core.windows.net",
            credential=self.azure_credential,
        )
        blob_container = blob_service.get_container_client(self.azure_storage_container)

        if not blob_container.exists():
            logging.info(
                f"Creating blob container {self.azure_storage_container} in storage account {self.azure_storage_container}"
            )
            blob_container.create_container()

        logging.info(f"Uploading files to Blob container...")
        if os.path.isfile(self.file_path):
            # If the path is a file, upload it directly
            blob_name = (
                f"{save_path}/{os.path.basename(self.file_path)}"
                if save_path
                else os.path.basename(self.file_path)
            )
            with open(self.file_path, "rb") as data:
                blob_container.upload_blob(name=blob_name, data=data, overwrite=True)
        elif os.path.isdir(self.file_path):
            for root, dirs, files in os.walk(self.file_path):
                for file in files:
                    blob_name = f"{save_path}/{file}" if save_path else file
                    with open(os.path.join(root, file), "rb") as data:
                        blob_container.upload_blob(
                            name=blob_name, data=data, overwrite=True
                        )
        logging.info(f"Files uploaded to Blob container {self.azure_storage_container}")

    def embed_query(self, query):
        openai.api_base = self.openai_endpoint
        openai.api_version = self.openai_api_version
        openai.api_type = self.openai_api_type
        openai.api_key = self.openai_apikey
        response = openai.Embedding.create(
            input=query, engine=self.openai_deploymentname
        )
        return response["data"][0]["embedding"]


class AISearchIndexer(CreateIndex):
    def __init__(
        self,
        file_path,
        azure_credentials,
        openai_apikey,
        openai_endpoint,
        openai_deploymentname,
        azure_storage_account,
        azure_storage_container,
        endpoint,
        api_key,
        api_version,
        blob_connection_string,
        data_source_name,
        search_index_name,
        vector_index_name,
        indexer_name,
        vector_skillset_name,
        embedding_dims,
        name_suffix="",
    ) -> None:
        super().__init__(
            file_path=file_path,
            azure_credentials=azure_credentials,
            openai_apikey=openai_apikey,
            openai_endpoint=openai_endpoint,
            openai_deploymentname=openai_deploymentname,
            azure_storage_account=azure_storage_account,
            azure_storage_container=azure_storage_container,
        )
        self.endpoint = endpoint
        self.api_key = api_key
        self.api_version = api_version
        self.headers = {"Content-Type": "application/json", "api-key": self.api_key}
        self.blob_connection_string = blob_connection_string
        self.data_source_name = data_source_name
        self.search_index_name = search_index_name
        self.vector_index_name = vector_index_name
        self.indexer_name = indexer_name
        self.vector_skillset_name = vector_skillset_name
        self.max_service_name_size = 28
        self.embedding_dims = embedding_dims
        self.service_name_suffix = name_suffix
        self.synonym_map = self.generate_service_name("synonym-map")
        self.vector_search_profile = self.generate_service_name("vector-profile")
        self.vector_search_config = self.generate_service_name("vector-search-config")
        self.vector_search_vectorizer = self.generate_service_name("vectorizer")
        self.semantic_config = self.generate_service_name("semantic-config")

    def generate_service_name(self, service_name_prefix):
        # Generate a UUID
        # uuid_str = str(uuid.uuid4())

        # Concatenate the prefix and the UUID
        service_name = service_name_prefix + "-" + self.service_name_suffix

        # Truncate the service name to the maximum size if necessary
        if len(service_name) > self.max_service_name_size:
            service_name = service_name[: self.max_service_name_size]

        return service_name

    def create_data_source_blob_storage(
        self,
        query=None,
    ) -> bool:
        query = "" if query is None else query
        data_source_payload = {
            "name": self.data_source_name,
            "description": "Data source for Azure Blob storage container",
            "type": "azureblob",
            "credentials": {"connectionString": self.blob_connection_string},
            "container": {"name": self.azure_storage_container, "query": query},
            "dataChangeDetectionPolicy": None,
            "dataDeletionDetectionPolicy": None,
        }

        response = requests.put(
            f"{self.endpoint}/datasources('{self.data_source_name}')?api-version={self.api_version}",
            headers=self.headers,
            json=data_source_payload,
        )
        if response.status_code in [200, 201, 204]:
            # self.data_source = response.json()
            return True
        else:
            logging.error(f"ERROR: {response.json()}")
            logging.error(f"ERROR: {response.status_code}")
            return False

    def check_index_exists(self, index_name):
        response = requests.get(
            f"{self.endpoint}/indexes('{index_name}')?api-version={self.api_version}",
            headers=self.headers,
        )
        return response.status_code == 200

    def check_indexer_exists(self):
        response = requests.get(
            f"{self.endpoint}/indexers('{self.indexer_name}')?api-version={self.api_version}",
            headers=self.headers,
        )
        return response.status_code == 200

    def create_skillset(self, model_uri, model_name, model_api_key):
        """
        Create a skillset for the indexer
        This skillset will be used to enrich the content before indexing
        """
        skillset_payload = {
            "name": self.vector_skillset_name,
            "description": "skills required for vector embedding creation processing",
            "skills": [
                {
                    "@odata.type": "#Microsoft.Skills.Util.DocumentExtractionSkill",
                    "parsingMode": "default",
                    "dataToExtract": "contentAndMetadata",
                    "configuration": {
                        "imageAction": "none",
                    },
                    "context": "/document",
                    "inputs": [{"name": "file_data", "source": "/document/file_data"}],
                    "outputs": [{"name": "content", "targetName": "content"}],
                },
                {
                    "@odata.type": "#Microsoft.Skills.Text.SplitSkill",
                    "name": "text-chunking-skill",
                    "description": "Skillset to describe the Text chunking required for vectorization",
                    "context": "/document",
                    "defaultLanguageCode": "en",
                    "textSplitMode": "pages",
                    "maximumPageLength": 2000,
                    "pageOverlapLength": 500,
                    "maximumPagesToTake": 0,
                    "inputs": [{"name": "text", "source": "/document/content"}],
                    "outputs": [{"name": "textItems", "targetName": "chunks"}],
                },
                {
                    "@odata.type": "#Microsoft.Skills.Text.AzureOpenAIEmbeddingSkill",
                    "name": "embedding-generation-skill",
                    "description": "",
                    "context": "/document/chunks/*",
                    "resourceUri": model_uri,
                    "apiKey": model_api_key,
                    "deploymentId": model_name,
                    "inputs": [{"name": "text", "source": "/document/chunks/*"}],
                    "outputs": [{"name": "embedding", "targetName": "embedding"}],
                },
            ],
            "indexProjections": {
                "selectors": [
                    {
                        "targetIndexName": self.vector_index_name,
                        "parentKeyFieldName": "parent_key",
                        "sourceContext": "/document/chunks/*",
                        "mappings": [
                            {
                                "name": "chunk",
                                "source": "/document/chunks/*",
                                "sourceContext": None,
                                "inputs": [],
                            },
                            {
                                "name": "embedding",
                                "source": "/document/chunks/*/embedding",
                                "sourceContext": None,
                                "inputs": [],
                            },
                        ],
                    }
                ],
            },
        }

        response = requests.put(
            f"{self.endpoint}/skillsets('{self.vector_skillset_name}')?api-version={self.api_version}",
            headers=self.headers,
            json=skillset_payload,
        )
        if response.status_code in [200, 201, 204]:
            return True
        else:
            logging.error(f"ERROR: {response.status_code}")
            return False

    def create_index(
        self,
        index_name,
        schema,
        scoring_profile=[],
        vector_search_config=None,
        semantic_config=None,
        rebuild=False,
    ):
        if self.check_index_exists(index_name):
            if rebuild:
                self.delete_index(index_name)
            else:
                return True
        payload = {
            "name": index_name,
            "defaultScoringProfile": "",
            "fields": schema,
            "scoringProfiles": scoring_profile,
            "similarity": {
                "@odata.type": "#Microsoft.Azure.Search.BM25Similarity",
                "k1": None,
                "b": None,
            },
            "semantic": semantic_config,
            "vectorSearch": vector_search_config,
        }
        response = requests.put(
            f"{self.endpoint}/indexes('{index_name}')?api-version={self.api_version}",
            headers=self.headers,
            json=payload,
        )
        if response.status_code in [200, 201, 204]:
            return True
        else:
            logging.error(f"ERROR: {response.status_code}|| {response.text}")
            return False

    def delete_index(self, index_name):
        response = requests.delete(
            f"{self.endpoint}/indexes('{index_name}')?api-version={self.api_version}",
            headers=self.headers,
        )
        logging.info(f"DELETED: {index_name}, {response.status_code}")
        return response.status_code == 204

    def get_vector_search_config(
        self,
        model_uri,
        model_name,
        model_api_key,
        metric="cosine",
        m=4,
        efConstruction=400,
        efSearch=500,
    ):
        """
        Create a vector search configuration
        model_uri: the uri of the embedding model
        model_name: the deployment name of the embedding model
        model_api_key: the api key of the embedding model
        metric: the distance metric to use for the vector search, use cosine for OpenAI models
        m: bi-directional link count
        efConstruction: number of nearest neighbors to consider during indexiing
        efSearch: number of nearest neighbors to consider during search
        """
        config = {
            "algorithms": [
                {
                    "name": self.vector_search_config,
                    "kind": "hnsw",
                    "hnswParameters": {
                        "metric": metric,
                        "m": m,
                        "efConstruction": efConstruction,
                        "efSearch": efSearch,
                    },
                    "exhaustiveKnnParameters": None,
                }
            ],
            "profiles": [
                {
                    "name": self.vector_search_profile,
                    "algorithm": self.vector_search_config,
                    "vectorizer": self.vector_search_vectorizer,
                }
            ],
            "vectorizers": [
                {
                    "name": self.vector_search_vectorizer,
                    "kind": "azureOpenAI",
                    "azureOpenAIParameters": {
                        "resourceUri": model_uri,
                        "deploymentId": model_name,
                        "apiKey": model_api_key,
                        "authIdentity": None,
                    },
                    "customWebApiParameters": None,
                }
            ],
        }
        return config

    def get_semantic_config(self):
        config = {
            "defaultConfiguration": None,
            "configurations": [
                {
                    "name": self.semantic_config,
                    "prioritizedFields": {
                        "titleField": None,
                        "prioritizedContentFields": [{"fieldName": "chunk"}],
                        "prioritizedKeywordsFields": [
                            {"fieldName": "id"},
                            {"fieldName": "parent_key"},
                        ],
                    },
                }
            ],
        }
        return config

    def get_schema(self, file_extension="txt", index_type="text"):
        if file_extension in ["txt", "pdf"] and index_type == "text":
            schema = [
                {
                    "name": "id",
                    "type": "Edm.String",
                    "key": True,
                    "searchable": False,
                    "filterable": True,
                    "sortable": False,
                    "facetable": False,
                },
                {
                    "name": "metadata_storage_name",
                    "type": "Edm.String",
                    "retrievable": True,
                    "searchable": False,
                    "filterable": True,
                    "sortable": False,
                },
                {
                    "name": "content",
                    "type": "Edm.String",
                    "retrievable": True,
                    "searchable": True,
                    "filterable": False,
                    "sortable": False,
                    "facetable": False,
                },
            ]
        elif file_extension in ["txt", "pdf"] and index_type == "vector":
            schema = [
                {
                    "name": "id",
                    "type": "Edm.String",
                    "key": True,
                    "searchable": True,
                    "filterable": False,
                    "sortable": False,
                    "facetable": False,
                    "analyzer": "keyword",
                },
                {
                    "name": "chunk",
                    "type": "Edm.String",
                    "retrievable": True,
                    "searchable": True,
                    "filterable": False,
                    "sortable": False,
                    "facetable": False,
                    "key": False,
                    "analyzer": "standard.lucene",
                },
                {
                    "name": "parent_key",
                    "type": "Edm.String",
                    "retrievable": True,
                    "searchable": False,
                    "filterable": True,
                    "sortable": False,
                    "facetable": False,
                    "key": False,
                },
                {
                    "name": "embedding",
                    "type": "Collection(Edm.Single)",
                    "retrievable": False,
                    "searchable": True,
                    "filterable": False,
                    "sortable": False,
                    "facetable": False,
                    "dimensions": self.embedding_dims,
                    "vectorSearchProfile": self.vector_search_profile,
                },
            ]

        return schema

    def create_indexer(
        self,
        cache_storage_connection,
        parsing_mode="default",
        disable_at_creation=False,
        batch_size=1,
        max_failed_items=100,
        output_field_mapping=[],
    ):
        """
        Create an indexer to index the data source
        cache_storage_connection: connection string to the storage account for caching
        parsing_mode: the mode to use for parsing the data source, "text", "delimitedText","json","jsonArray","jsonLines"
        """
        if self.check_index_exists(self.search_index_name):
            indexer_payload = {
                "name": self.indexer_name,
                "description": "Indexer for Azure Blob storage container",
                "dataSourceName": self.data_source_name,
                "targetIndexName": self.search_index_name,
                "skillsetName": self.vector_skillset_name,
                "disabled": disable_at_creation,
                "parameters": {
                    "configuration": {
                        "parsingMode": parsing_mode,
                        "dataToExtract": "contentAndMetadata",
                    },
                    "batchSize": batch_size,
                    "maxFailedItems": max_failed_items,
                },
                "outputFieldMappings": output_field_mapping,
                "cache": {
                    "enableReprocessing": True,
                    "storageConnectionString": cache_storage_connection,
                },
            }
            response = requests.put(
                f"{self.endpoint}/indexers('{self.indexer_name}')?api-version={self.api_version}",
                headers=self.headers,
                json=indexer_payload,
            )
            if response.status_code in [200, 201, 204]:
                # self.indexer = response.json()
                return True
            else:
                logging.error(f"ERROR: {response.status_code}, {response.text}")
                return False
        else:
            return False

    def run_indexer(self, reset_flag=False):
        if self.check_indexer_exists():
            indexer_payload = {
                "x-ms-client-request-id": str(uuid.uuid4()),
            }
            if reset_flag:
                response = requests.post(
                    f"{self.endpoint}/indexers('{self.indexer_name}')/search.reset?api-version={self.api_version}",
                    headers=self.headers,
                    json=indexer_payload,
                )
                assert response.status_code == 204, "Indexer reset failed."
            response = requests.post(
                f"{self.endpoint}/indexers('{self.indexer_name}')/search.run?api-version={self.api_version}",
                headers=self.headers,
                json=indexer_payload,
            )
            if response.status_code in [202]:
                return True
            else:
                logging.error(f"{response.status_code}")
                return False

    def get_indexer_status(self):
        response = requests.get(
            f"{self.endpoint}/indexers('{self.indexer_name}')/status?api-version={self.api_version}",
            headers=self.headers,
        )
        if response.status_code == 200:
            return response.json()["lastResult"]
        else:
            logging.error(f"ERROR: {response.status_code}")
            return None

    def log_indexer_status(self, interval=60, retry_count=10):
        results = self.get_indexer_status()
        status = results["status"]
        if status == "inProgress" and retry_count > 0:
            logging.info(f"Indexer status: {status}. Retrying in {interval} seconds.")
            time.sleep(interval)
            self.log_indexer_status(retry_count=(retry_count - 1))
        else:
            time_taken = get_time_difference_in_minutes(
                results["startTime"], results["endTime"]
            )
            logging.info(f"Indexer status: {status}")
            logging.info(f"Total documents: {results['itemsProcessed']}")
            logging.info(f"Failed documents: {results['itemsFailed']}")
            logging.info(f"Total time: {time_taken:.2f} minutes")


def get_time_difference_in_minutes(start_time_str, end_time_str):
    # Parse the UTC time strings into datetime objects
    start_time = datetime.strptime(start_time_str, "%Y-%m-%dT%H:%M:%S.%fZ")
    end_time = datetime.strptime(end_time_str, "%Y-%m-%dT%H:%M:%S.%fZ")

    # Calculate the difference in time
    time_diff = end_time - start_time

    # Convert the time difference to minutes
    time_diff_in_minutes = time_diff.total_seconds() / 60

    return time_diff_in_minutes
