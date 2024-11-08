import argparse
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import os
from collections import Counter
import requests
import uuid
from datetime import datetime
import time
import openai
from azure.identity import DefaultAzureCredential
import dotenv
import logging

dotenv.load_dotenv()

EMBEDDING_DIMS=1536
logging.basicConfig(level=logging.INFO)
AZURE_OPENAI_SERVICE = os.environ.get("AZURE_OPENAI_SERVICE")
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT_NAME = (
    os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME") or "embedding"
)
AZURE_SEARCH_SERVICE_ENDPOINT = os.environ.get("AZURE_SEARCH_SERVICE_ENDPOINT")
AZURE_SEARCH_SERVICE_API_KEY = os.environ.get("AZURE_SEARCH_SERVICE_API_KEY")
AZURE_STORAGE_ACCOUNT = os.environ.get("AZURE_STORAGE_ACCOUNT")
AZURE_STORAGE_CONTAINER = os.environ.get("AZURE_STORAGE_CONTAINER")
AZURE_STORAGE_CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
open_ai_token_cache = {}
CACHE_KEY_TOKEN_CRED = "openai_token_cred"
CACHE_KEY_CREATED_TIME = "created_time"

def get_api_key_token(azure_credential, scope):
    return azure_credential.get_token(scope).token

def get_aoai_service(azure_credential):
    openai.api_type = "azure"
    openai.api_base = f"https://{AZURE_OPENAI_SERVICE}.openai.azure.com"
    openai.api_version = "2023-10-01"
    openai.api_type = "azure_ad"
    openai.api_key = get_api_key_token(azure_credential, scope="https://cognitiveservices.azure.com/.default")
    open_ai_token_cache[CACHE_KEY_CREATED_TIME] = time.time()
    open_ai_token_cache[CACHE_KEY_TOKEN_CRED] = azure_credential
    
    return openai.api_base, AZURE_OPENAI_API_KEY


class AISearchIndexer:

    def __init__(self,
                endpoint, 
                api_key, 
                api_version, 
                data_source_name,
                search_index_name,
                vector_index_name,
                indexer_name,
                vector_skillset_name,
                embedding_dims=EMBEDDING_DIMS,
                name_suffix="",
                ) -> None:
        self.endpoint = endpoint
        self.api_key = api_key
        self.api_version = api_version
        self.headers = {"Content-Type": "application/json", "api-key": self.api_key}
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
        self, blob_connection, blob_container_name, query
    ) -> bool:
        query = "" if query is None else query
        data_source_payload = {
            "name": self.data_source_name,
            "description": "Data source for Azure Blob storage container",
            "type": "azureblob",
            "credentials": {"connectionString": blob_connection},
            "container": {"name": blob_container_name, "query": query},
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
                        "configuration": {"imageAction": "none",},
                        "context": "/document",
                        "inputs": [
                        {
                            "name": "file_data",
                            "source": "/document/file_data"
                        }
                        ],
                        "outputs": [
                        {
                            "name": "content",
                            "targetName": "content"
                        }
                        ]
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
    
    def create_index(self, index_name, schema, scoring_profile = [], vector_search_config=None, semantic_config=None, rebuild = False,):
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
    
    def get_vector_search_config(self, model_uri, model_name, model_api_key, metric="cosine", m=4, efConstruction=400, efSearch=500):
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
    
    def get_schema(self, file_extension = "txt", index_type="text"):
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
    
    def create_indexer(self, 
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
                "disabled" : disable_at_creation,
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
            return response.json()['lastResult']
        else:
            logging.error(f"ERROR: {response.status_code}")
            return None
        
    def log_indexer_status(self, interval=60, retry_count=10):
        results = self.get_indexer_status()
        status = results['status']
        if status =="inProgress" and retry_count>0:
            logging.info(f"Indexer status: {status}. Retrying in {interval} seconds.")  
            time.sleep(interval)
            self.log_indexer_status(retry_count=(retry_count - 1))
        else:
            time_taken = get_time_difference_in_minutes(results['startTime'], results['endTime'])
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

def build_search_index(azure_credential, save_path, model_uri, model_name, model_api_key, vector_flag, semantic_flag, file_extension, run_indexer, reset_indexer):
    # Create the AI Search Indexer
    logging.info(f"STARTING indexing of the crawled data. {AZURE_SEARCH_SERVICE_ENDPOINT}")
    try:
        search_indexer = AISearchIndexer(
            endpoint=AZURE_SEARCH_SERVICE_ENDPOINT, 
            api_key=AZURE_SEARCH_SERVICE_API_KEY,
            api_version="2023-10-01-Preview",z
            data_source_name=f"{save_path}-data-source",
            search_index_name=f"{save_path}-search-index",
            vector_index_name=f"{save_path}-vector-index",
            indexer_name=f"{save_path}-indexer",
            vector_skillset_name=f"{save_path}vectorskillset".replace("-", ""),
            name_suffix="1234",
        )
        # Step 1 - Create the Data Source
        response = search_indexer.create_data_source_blob_storage(
            blob_connection=AZURE_STORAGE_CONNECTION_STRING,
            blob_container_name=AZURE_STORAGE_CONTAINER,
            query=save_path,
        )
        logging.info(f"Search Data Source status = {response}.")

        # Step 2 - Create vector and semantic config if required
        text_schema = search_indexer.get_schema(file_extension="txt", index_type="text")
        if vector_flag:
            vector_search_config = search_indexer.get_vector_search_config(
                model_uri=model_uri,
                model_name=model_name,
                model_api_key=model_api_key,
            )
            vector_schema = search_indexer.get_schema(file_extension=file_extension, index_type="vector")
        if semantic_flag:
            semantic_config = search_indexer.get_semantic_config()

        
        # Step 3 - Create the Search Index
        response = search_indexer.create_index(index_name = search_indexer.search_index_name, schema=text_schema, scoring_profile = [], vector_search_config=None, semantic_config=None, rebuild = False,)
        logging.info(f"Keyword Search Index status = {response}.")
        if vector_flag:
            response = search_indexer.create_index(index_name = search_indexer.vector_index_name, schema=vector_schema, scoring_profile = [], vector_search_config=vector_search_config, semantic_config=semantic_config, rebuild = False,)
            logging.info(f"Vector Search Index status = {response}.")
        
        # Step 4 - Create the Vector embedding skillset to enhance the indexer
        response = search_indexer.create_skillset(
            model_uri=model_uri,
            model_name=model_name,
            model_api_key=model_api_key,
        )
        logging.info(f"Vector Skillset status = {response}.")

        # Step 5 - Create the indexer which will ultimately call the vector embedding skillset
        response = search_indexer.create_indexer(
            cache_storage_connection=AZURE_STORAGE_CONNECTION_STRING,
            disable_at_creation=not(run_indexer),
        )
        logging.info(f"Search Indexer status = {response}")

        # Step 6 - Run the indexer, if config is set to True
        if run_indexer:
            response = search_indexer.run_indexer(reset_flag=reset_indexer)
            logging.info(f"Search Indexer Run status = {response}")
        
        # Log the status
        search_indexer.log_indexer_status()
        
    except Exception as e:
        logging.error(
            "Creation of Search index has Failed! Error: %s", e
        )
        return False
    

def most_common_extension(folder_path):
    # Get list of all files in directory
    files = os.listdir(folder_path)

    # Extract file extensions and count occurrences
    extensions = Counter(file.split('.')[-1] for file in files if '.' in file)

    # Find and return the most common extension
    return extensions.most_common(1)[0] if extensions else None


def upload_files_to_blob(azure_credential, folder_path, save_path=""):
    blob_service = BlobServiceClient(
        account_url=f"https://{AZURE_STORAGE_ACCOUNT}.blob.core.windows.net",
        credential=azure_credential,
    )
    blob_container = blob_service.get_container_client(AZURE_STORAGE_CONTAINER)

    if not blob_container.exists():
        logging.info(
            f"Creating blob container {AZURE_STORAGE_CONTAINER} in storage account {AZURE_STORAGE_ACCOUNT}"
        )
        blob_container.create_container()

    logging.info(f"Uploading files to Blob container...")
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            blob_name = f"{save_path}/{file}" if save_path else file
            with open(os.path.join(root, file), "rb") as data:
                blob_container.upload_blob(name=blob_name, data=data, overwrite=True)
    logging.info(f"Files uploaded to Blob container {AZURE_STORAGE_CONTAINER}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder-path",
        type=str,
        default="data/text/",
        help="Path to the folder containing the text files to be uploaded",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="sample-text",
        help="Save file name for this text. It should be unique to this project",
    )
    parser.add_argument(
        "--vector-flag",
        type=bool,
        default=True,
        help="Flag to enable vector search",
    )
    parser.add_argument(
        "--semantic-flag",
        type=bool,
        default=True,
        help="Flag to enable semantic search",
    )
    parser.add_argument(
        "--run-indexer",
        type=bool,
        default=True,
        help="Flag to run the indexer after initialization",
    )
    parser.add_argument(
        "--reset-indexer",
        type=bool,
        default=True,
        help="Flag to reset the indexer before running it",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    azure_credential = DefaultAzureCredential(
        exclude_shared_token_cache_credential=True
    )
    folder_path = args.folder_path
    extension, count = most_common_extension(folder_path)
    upload_files_to_blob(azure_credential, folder_path=folder_path, save_path=args.save_path)
    model_uri, model_api_key = get_aoai_service(azure_credential)
    model_name =AZURE_OPENAI_DEPLOYMENT_NAME
    logging.info(f"Model URI: {model_uri}, extension: {extension}")
    build_search_index(
        azure_credential=azure_credential, 
        save_path=args.save_path,
        model_uri=model_uri, 
        model_name=model_name, 
        model_api_key=model_api_key, 
        vector_flag=args.vector_flag, 
        semantic_flag=args.semantic_flag, 
        file_extension=extension, 
        run_indexer=args.run_indexer, 
        reset_indexer=args.reset_indexer
    )
    

