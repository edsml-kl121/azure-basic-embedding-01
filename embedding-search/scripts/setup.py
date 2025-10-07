import asyncio
from azure.ai.ml import MLClient
from azure.core.credentials import AzureNamedKeyCredential
from azure.mgmt.storage.aio import StorageManagementClient
from azure.identity.aio import AzureCliCredential
import azure.identity
from azure.storage.blob.aio import BlobServiceClient
from azure.search.documents.indexes.aio import SearchIndexClient, SearchIndexerClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchIndexerSkillset,
    SearchIndexer,
    SearchIndexerDataSourceConnection,
    SearchIndexerDataContainer,
    SearchIndexerDataSourceType,
    SearchIndexerIndexProjections,
    SearchIndexerIndexProjectionSelector,
    SearchIndexerIndexProjectionsParameters,
    IndexProjectionMode,
    AzureMachineLearningSkill,
    SplitSkill,
    InputFieldMappingEntry,
    OutputFieldMappingEntry,
    SearchFieldDataType,
    ScalarQuantizationCompressionConfiguration,
    VectorSearchProfile,
    VectorSearch,
    SearchField,
    SearchableField,
    SimpleField,
    AzureOpenAIVectorizer,
    AzureOpenAIParameters,
    HnswAlgorithmConfiguration,
    LexicalAnalyzerName,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch
)
import os
import glob
from dotenv import load_dotenv

load_dotenv()  # This loads variables from .env file

current_file_directory = os.path.dirname(os.path.abspath(__file__))
samples_path = os.path.join(current_file_directory, "..", "..", "..", "..", "..", "data", "benefitdocs")

def create_credential():
    return AzureCliCredential(tenant_id=os.getenv("AZURE_TENANT_ID", None))

def create_sync_credential():
    return azure.identity.AzureCliCredential(tenant_id=os.getenv("AZURE_TENANT_ID", None))

async def main():
    async with create_credential() as credential:
        print("Uploading sample documents...")
        async with BlobServiceClient.from_connection_string(conn_str=await get_storage_connection_string()) as blob_service_client:
            await upload_documents(blob_service_client)

        print("Creating index...")
        search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        async with SearchIndexClient(endpoint=search_endpoint, credential=credential) as search_index_client:
            await create_index(search_index_client)
        
        async with SearchIndexerClient(endpoint=search_endpoint, credential=credential) as search_indexer_client:
            print("Creating skillset...")
            await create_skillset(search_indexer_client)
            print("Creating datasource...")
            await create_datasource(search_indexer_client)
            print("Creating indexer...")
            await create_indexer(search_indexer_client)

        print("Done")

async def upload_documents(blob_service_client: BlobServiceClient):
    container_client = blob_service_client.get_container_client(os.getenv("AZURE_STORAGE_CONTAINER"))
    document_paths = glob.glob(os.path.join(samples_path, "*.pdf"))
    for document_path in document_paths:
        async with container_client.get_blob_client(os.path.basename(document_path)) as blob_client:
            if not await blob_client.exists():
                with open(document_path, "rb") as data:
                    await blob_client.upload_blob(data=data)

async def get_storage_credential():
    async with create_credential() as credential, StorageManagementClient(credential=credential, subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID")) as storage_client:
        result = await storage_client.storage_accounts.list_keys(resource_group_name=os.getenv("AZURE_RESOURCE_GROUP"), account_name=os.getenv("AZURE_STORAGE_ACCOUNT"))
        return AzureNamedKeyCredential(name="key1", key=result.keys[0].value)

async def get_storage_connection_string():
    _, key = (await get_storage_credential()).named_key
    return f"DefaultEndpointsProtocol=https;AccountName={os.getenv('AZURE_STORAGE_ACCOUNT')};AccountKey={key};EndpointSuffix=core.windows.net;"

def get_openai_deployment():
    # Your Azure OpenAI configuration from the portal
    endpoint = "https://kandanail-0150-resource.openai.azure.com"
    api_key = "APIKEY"
    deployment_name = "text-embedding-3-large"
    return (endpoint, api_key, deployment_name)

async def create_index(search_index_client: SearchIndexClient):
    endpoint, api_key, deployment_name = get_openai_deployment()
    
    index = SearchIndex(
        name=os.getenv("AZURE_SEARCH_INDEX"),
        fields=[
            SearchableField(
                name="id",
                type=SearchFieldDataType.String,
                key=True,
                filterable=True,
                analyzer_name=LexicalAnalyzerName.KEYWORD
            ),
            SearchableField(
                name="document_id",
                type=SearchFieldDataType.String,
                key=False,
                filterable=True,
                analyzer_name=LexicalAnalyzerName.KEYWORD
            ),
            SearchField(
                name="content",
                type=SearchFieldDataType.String,
                searchable=True,
                filterable=False,
                facetable=False,
                sortable=False
            ),
            SearchField(
                name="embedding",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                stored=False,
                vector_search_dimensions=3072,  # text-embedding-3-large has 3072 dimensions
                vector_search_profile_name="approximateProfile"
            ),
            SimpleField(
                name="metadata_storage_path",
                type=SearchFieldDataType.String,
                filterable=True
            )
        ],
        vector_search=VectorSearch(
            profiles=[
                VectorSearchProfile(
                    name="approximateProfile",
                    algorithm_configuration_name="approximateConfiguration",
                    vectorizer="openai-vectorizer",
                    compression_configuration_name="scalarQuantization"
                )
            ],
            algorithms=[
                HnswAlgorithmConfiguration(name="approximateConfiguration")
            ],
            vectorizers=[
                AzureOpenAIVectorizer(
                    name="openai-vectorizer",
                    openai_parameters=AzureOpenAIParameters(
                        resource_uri=endpoint,
                        deployment_id=deployment_name,
                        api_key=api_key
                    )
                )
            ],
            compressions=[
                ScalarQuantizationCompressionConfiguration(name="scalarQuantization")
            ]
        ),
        semantic_search=SemanticSearch(
            default_configuration_name="semantic-config",
            configurations=[
                SemanticConfiguration(
                    name="semantic-config",
                    prioritized_fields=SemanticPrioritizedFields(
                        content_fields=[SemanticField(field_name="content")]
                    )
                )
            ]
        )
    )
    await search_index_client.create_or_update_index(index)

async def create_skillset(search_indexer_client: SearchIndexerClient):
    endpoint, api_key, deployment_name = get_openai_deployment()
    
    skillset = SearchIndexerSkillset(
        name=os.getenv("AZURE_SEARCH_SKILLSET"),
        skills=[
            AzureMachineLearningSkill(
                description="Skill to generate embeddings via Azure OpenAI",  
                context="/document/pages/*",
                scoring_uri=f"{endpoint}/openai/deployments/{deployment_name}/embeddings?api-version=2023-05-15",
                authentication_key=api_key,
                inputs=[  
                    InputFieldMappingEntry(name="input", source="=[$(/document/pages/*)]"),
                ],  
                outputs=[  
                    OutputFieldMappingEntry(name="data", target_name="embedding_data")  
                ]
            ),
            SplitSkill(  
                description="Split skill to chunk documents",  
                text_split_mode="pages",  
                context="/document",  
                maximum_page_length=2000,  
                page_overlap_length=500,  
                inputs=[  
                    InputFieldMappingEntry(name="text", source="/document/content"),  
                ],  
                outputs=[  
                    OutputFieldMappingEntry(name="textItems", target_name="pages")  
                ]
            )
        ],
        index_projections=SearchIndexerIndexProjections(
            selectors=[
                SearchIndexerIndexProjectionSelector(
                    target_index_name=os.getenv("AZURE_SEARCH_INDEX"),
                    parent_key_field_name="document_id",
                    source_context="/document/pages/*",
                    mappings=[
                        InputFieldMappingEntry(
                            name="embedding",
                            source="/document/pages/*/embedding_data/*/embedding"
                        ),
                        InputFieldMappingEntry(
                            name="content",
                            source="/document/pages/*"
                        ),
                        InputFieldMappingEntry(
                            name="metadata_storage_path",
                            source="/document/metadata_storage_path"
                        )
                    ]
                )
            ],
            parameters=SearchIndexerIndexProjectionsParameters(projection_mode=IndexProjectionMode.SKIP_INDEXING_PARENT_DOCUMENTS)
        )
    )
    await search_indexer_client.create_or_update_skillset(skillset)

async def create_datasource(search_indexer_client: SearchIndexerClient):
    datasource = SearchIndexerDataSourceConnection(
        name=os.getenv("AZURE_SEARCH_DATASOURCE"),
        type=SearchIndexerDataSourceType.AZURE_BLOB,
        connection_string=await get_storage_connection_string(),
        container=SearchIndexerDataContainer(name=os.getenv("AZURE_STORAGE_CONTAINER"))
    )
    await search_indexer_client.create_or_update_data_source_connection(datasource)

async def create_indexer(search_indexer_client: SearchIndexerClient):
    indexer = SearchIndexer(
        name=os.getenv("AZURE_SEARCH_INDEXER"),
        data_source_name=os.getenv("AZURE_SEARCH_DATASOURCE"),
        target_index_name=os.getenv("AZURE_SEARCH_INDEX"),
        skillset_name=os.getenv("AZURE_SEARCH_SKILLSET")
    )
    await search_indexer_client.create_or_update_indexer(indexer)

if __name__ == "__main__":
    asyncio.run(main())
