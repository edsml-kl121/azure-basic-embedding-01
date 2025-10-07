#!/usr/bin/env python
# coding: utf-8

# # Vector search in Python (Azure AI Search)
# 
# This code demonstrates how to use Azure AI Search by using the push API to insert vectors into your search index:
# 
# + Create an index schema
# + Load the sample data from a local folder
# + Embed the documents in-memory using Azure OpenAI's text-embedding-3-large model
# + Index the vector and nonvector fields on Azure AI Search
# + Run a series of vector and hybrid queries, including metadata filtering and hybrid (text + vectors) search. 
# 
# The code uses Azure OpenAI to generate embeddings for title and content fields. You'll need access to Azure OpenAI to run this demo.
# 
# The code reads the `text-sample.json` file, which contains the input data for which embeddings need to be generated.
# 
# The output is a combination of human-readable text and embeddings that can be pushed into a search index.
# 
# ## Prerequisites
# 
# + An Azure subscription, with [access to Azure OpenAI](https://aka.ms/oai/access). You must have the Azure OpenAI service name and an API key.
# 
# + A deployment of the text-embedding-3-large embedding model.
# 
# + Azure AI Search, any tier, but choose a service that has sufficient capacity for your vector index. We recommend Basic or higher. [Enable semantic ranking](https://learn.microsoft.com/azure/search/semantic-how-to-enable-disable) if you want to run the hybrid query with semantic ranking.
# 
# We used Python 3.11, [Visual Studio Code with the Python extension](https://code.visualstudio.com/docs/python/python-tutorial), and the [Jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) to test this example.

# ### Set up a Python virtual environment in Visual Studio Code
# 
# 1. Open the Command Palette (Ctrl+Shift+P).
# 1. Search for **Python: Create Environment**.
# 1. Select **Venv**.
# 1. Select a Python interpreter. Choose 3.10 or later.
# 
# It can take a minute to set up. If you run into problems, see [Python environments in VS Code](https://code.visualstudio.com/docs/python/environments).

# ### Install packages

# In[10]:

from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
import os

load_dotenv(override=True) # take environment variables from .env.

# The following variables from your .env file are used in this notebook
endpoint = os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"]
credential = AzureKeyCredential(os.getenv("AZURE_SEARCH_ADMIN_KEY", "")) if len(os.getenv("AZURE_SEARCH_ADMIN_KEY", "")) > 0 else DefaultAzureCredential()
index_name = os.getenv("AZURE_SEARCH_INDEX", "vectest")
azure_openai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
azure_openai_key = os.getenv("AZURE_OPENAI_KEY", "") if len(os.getenv("AZURE_OPENAI_KEY", "")) > 0 else None
azure_openai_embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
azure_openai_embedding_dimensions = int(os.getenv("AZURE_OPENAI_EMBEDDING_DIMENSIONS", 1024))
embedding_model_name = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")


# ## Create embeddings
# Read your data, generate OpenAI embeddings and export to a format to insert your Azure AI Search index:

# In[14]:


from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
import json

openai_credential = DefaultAzureCredential()
token_provider = get_bearer_token_provider(openai_credential, "https://cognitiveservices.azure.com/.default")

client = AzureOpenAI(
    azure_deployment=azure_openai_embedding_deployment,
    api_version=azure_openai_api_version,
    azure_endpoint=azure_openai_endpoint,
    api_key=azure_openai_key,
    azure_ad_token_provider=token_provider if not azure_openai_key else None
)

# Generate Document Embeddings using OpenAI 3 large
# Read the text-sample.json
path = "text-sample.json"
with open(path, 'r', encoding='utf-8') as file:
    input_data = json.load(file)

titles = [item['title'] for item in input_data]
content = [item['content'] for item in input_data]
title_response = client.embeddings.create(input=titles, model=embedding_model_name, dimensions=azure_openai_embedding_dimensions)
title_embeddings = [item.embedding for item in title_response.data]
content_response = client.embeddings.create(input=content, model=embedding_model_name, dimensions=azure_openai_embedding_dimensions)
content_embeddings = [item.embedding for item in content_response.data]

# Generate embeddings for title and content fields
for i, item in enumerate(input_data):
    title = item['title']
    content = item['content']
    item['titleVector'] = title_embeddings[i]
    item['contentVector'] = content_embeddings[i]

# Output embeddings to docVectors.json file
output_path = os.path.join('output', 'docVectors.json')
output_directory = os.path.dirname(output_path)
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
with open(output_path, "w") as f:
    json.dump(input_data, f)


# ## Create your search index
# 
# Create your search index schema and vector search configuration. If you get an error, check the search service for available quota and check the .env file to make sure you're using a unique search index name.

# In[17]:


from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SimpleField,
    SearchFieldDataType,
    SearchableField,
    SearchField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    SemanticSearch,
    SearchIndex,
    AzureOpenAIVectorizer,
    AzureOpenAIVectorizerParameters
)


# Create a search index
index_client = SearchIndexClient(
    endpoint=endpoint, credential=credential)
fields = [
    SimpleField(name="id", type=SearchFieldDataType.String, key=True, sortable=True, filterable=True, facetable=True),
    SearchableField(name="title", type=SearchFieldDataType.String),
    SearchableField(name="content", type=SearchFieldDataType.String),
    SearchableField(name="category", type=SearchFieldDataType.String,
                    filterable=True),
    SearchField(name="titleVector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True, vector_search_dimensions=azure_openai_embedding_dimensions, vector_search_profile_name="myHnswProfile"),
    SearchField(name="contentVector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True, vector_search_dimensions=azure_openai_embedding_dimensions, vector_search_profile_name="myHnswProfile"),
]

# Configure the vector search configuration  
vector_search = VectorSearch(
    algorithms=[
        HnswAlgorithmConfiguration(
            name="myHnsw"
        )
    ],
    profiles=[
        VectorSearchProfile(
            name="myHnswProfile",
            algorithm_configuration_name="myHnsw",
            vectorizer_name="myVectorizer"
        )
    ],
    vectorizers=[
        AzureOpenAIVectorizer(
            vectorizer_name="myVectorizer",
            parameters=AzureOpenAIVectorizerParameters(
                resource_url=azure_openai_endpoint,
                deployment_name=azure_openai_embedding_deployment,
                model_name=embedding_model_name,
                api_key=azure_openai_key
            )
        )
    ]
)



semantic_config = SemanticConfiguration(
    name="my-semantic-config",
    prioritized_fields=SemanticPrioritizedFields(
        title_field=SemanticField(field_name="title"),
        keywords_fields=[SemanticField(field_name="category")],
        content_fields=[SemanticField(field_name="content")]
    )
)

# Create the semantic settings with the configuration
semantic_search = SemanticSearch(configurations=[semantic_config])

# Create the search index with the semantic settings
index = SearchIndex(name=index_name, fields=fields,
                    vector_search=vector_search, semantic_search=semantic_search)
result = index_client.create_or_update_index(index)
print(f'{result.name} created')


# In[19]:


result


# ## Insert text and embeddings into vector store
# Add texts and metadata from the JSON data to the vector store:

# In[18]:


from azure.search.documents import SearchClient
import json

# Upload some documents to the index
output_path = os.path.join('output', 'docVectors.json')
output_directory = os.path.dirname(output_path)
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
with open(output_path, 'r') as file:  
    documents = json.load(file)  
search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)
result = search_client.upload_documents(documents)
print(f"Uploaded {len(documents)} documents") 


# If you are indexing a very large number of documents, you can use the `SearchIndexingBufferedSender` which is an optimized way to automatically index the docs as it will handle the batching for you:

# In[46]:


from azure.search.documents import SearchIndexingBufferedSender

# Upload some documents to the index  
with open(output_path, 'r') as file:  
    documents = json.load(file)  

# Use SearchIndexingBufferedSender to upload the documents in batches optimized for indexing  
with SearchIndexingBufferedSender(  
    endpoint=endpoint,  
    index_name=index_name,  
    credential=credential,  
) as batch_client:  
    # Add upload actions for all documents  
    batch_client.upload_documents(documents=documents)  
print(f"Uploaded {len(documents)} documents in total")  


# In[47]:


# Helper code to print results

from azure.search.documents import SearchItemPaged

def print_results(results: SearchItemPaged[dict]):
    semantic_answers = results.get_answers()
    if semantic_answers:
        for answer in semantic_answers:
            if answer.highlights:
                print(f"Semantic Answer: {answer.highlights}")
            else:
                print(f"Semantic Answer: {answer.text}")
            print(f"Semantic Answer Score: {answer.score}\n")

    for result in results:
        print(f"Title: {result['title']}")  
        print(f"Score: {result['@search.score']}")
        if result.get('@search.reranker_score'):
            print(f"Reranker Score: {result['@search.reranker_score']}")
        print(f"Content: {result['content']}")  
        print(f"Category: {result['category']}\n")

        captions = result["@search.captions"]
        if captions:
            caption = captions[0]
            if caption.highlights:
                print(f"Caption: {caption.highlights}\n")
            else:
                print(f"Caption: {caption.text}\n")


# ## Perform a vector similarity search
# 
# This example shows a pure vector search using a pre-computed embedding

# In[48]:


from azure.search.documents.models import VectorizedQuery

# Pure Vector Search
query = "tools for software development"  
embedding = client.embeddings.create(input=query, model=embedding_model_name, dimensions=azure_openai_embedding_dimensions).data[0].embedding

# 50 is an optimal value for k_nearest_neighbors when performing vector search
# To learn more about how vector ranking works, please visit https://learn.microsoft.com/azure/search/vector-search-ranking
vector_query = VectorizedQuery(vector=embedding, k_nearest_neighbors=50, fields="contentVector")

results = search_client.search(  
    search_text=None,  
    vector_queries= [vector_query],
    select=["title", "content", "category"],
    top=3
)  

print_results(results)


# ## Perform a vector similarity search using a vectorizable text query

# This example shows a pure vector search using the vectorizable text query, all you need to do is pass in text and your vectorizer will handle the query vectorization.

# In[49]:


from azure.search.documents.models import VectorizableTextQuery

# Pure Vector Search
query = "tools for software development"  

vector_query = VectorizableTextQuery(text=query, k_nearest_neighbors=50, fields="contentVector")

results = search_client.search(  
    search_text=None,  
    vector_queries= [vector_query],
    select=["title", "content", "category"],
    top=3
)  

print_results(results)


# This example shows a pure vector search that demonstrates Azure OpenAI's text-embedding-3-large multilingual capabilities. 

# In[18]:


# Pure Vector Search multi-lingual (e.g 'tools for software development' in Dutch)  
query = "tools voor softwareontwikkeling"  

vector_query = VectorizableTextQuery(text=query, k_nearest_neighbors=50, fields="contentVector")

results = search_client.search(  
    search_text=None,  
    vector_queries= [vector_query],
    select=["title", "content", "category"],
    top=3
)  

print_results(results)



# ## Perform an Exhaustive KNN exact nearest neighbor search

# This example shows how you can exhaustively search your vector index regardless of what index you have, HNSW or ExhaustiveKNN. You can use this to calculate the ground-truth values.

# In[19]:


# Pure Vector Search
query = "tools for software development"  

vector_query = VectorizableTextQuery(text=query, k_nearest_neighbors=50, fields="contentVector", exhaustive=True)

results = search_client.search(  
    search_text=None,  
    vector_queries= [vector_query],
    select=["title", "content", "category"],
    top=3
)  

print_results(results)


# ## Perform a Cross-Field Vector Search
# 
# This example shows a cross-field vector search that allows you to query multiple vector fields at the same time. Note, ensure that the same embedding model was used for the vector fields you decide to query.

# In[20]:


# Pure Vector Search
query = "tools for software development"  

vector_query = VectorizableTextQuery(text=query, k_nearest_neighbors=50, fields="contentVector, titleVector")

results = search_client.search(  
    search_text=None,  
    vector_queries= [vector_query],
    select=["title", "content", "category"],
    top=3
)  

print_results(results)


# ## Perform a Multi-Vector Search
# 
# This example shows a cross-field vector search that allows you to query multiple vector fields at the same time by passing in multiple query vectors. Note, in this case, you can pass in query vectors from two different embedding models to the corresponding vector fields in your index.

# In[21]:


# Multi-Vector Search
query = "tools for software development"  


vector_query_1 = VectorizableTextQuery(text=query, k_nearest_neighbors=50, fields="titleVector")
vector_query_2 = VectorizableTextQuery(text=query, k_nearest_neighbors=50, fields="contentVector")

results = search_client.search(  
    search_text=None,  
    vector_queries=[vector_query_1, vector_query_2],
    select=["title", "content", "category"],
    top=3
)  

print_results(results)


# ## Perform a weighted multi-vector search
# 
# This example shows a cross-field vector search that allows you to query multiple vector fields at the same time by passing in multiple query vectors with different weighting

# In[22]:


# Multi-Vector Search
query = "tools for software development"  


vector_query_1 = VectorizableTextQuery(text=query, k_nearest_neighbors=50, fields="titleVector", weight=2)
vector_query_2 = VectorizableTextQuery(text=query, k_nearest_neighbors=50, fields="contentVector", weight=0.5)

results = search_client.search(  
    search_text=None,  
    vector_queries=[vector_query_1, vector_query_2],
    select=["title", "content", "category"],
    top=3
)  

print_results(results)


# ## Perform a Pure Vector Search with a filter
# This example shows how to apply filters on your index. Note, that you can choose whether you want to use Pre-Filtering (default) or Post-Filtering.

# In[23]:


from azure.search.documents.models import VectorFilterMode

# Pure Vector Search
query = "tools for software development"  

vector_query = VectorizableTextQuery(text=query, k_nearest_neighbors=50, fields="contentVector")

results = search_client.search(  
    search_text=None,  
    vector_queries= [vector_query],
    vector_filter_mode=VectorFilterMode.PRE_FILTER,
    filter="category eq 'Developer Tools'",
    select=["title", "content", "category"],
    top=3
)

print_results(results)


# ## Perform a Hybrid Search

# In[24]:


# Hybrid Search
query = "scalable storage solution"  

vector_query = VectorizableTextQuery(text=query, k_nearest_neighbors=50, fields="contentVector")

results = search_client.search(  
    search_text=query,  
    vector_queries=[vector_query],
    select=["title", "content", "category"],
    top=3
)  

print_results(results)


# ## Perform a weighted hybrid search
# 
# Change the weighting of the vector query compared to the text query

# In[25]:


# Hybrid Search
query = "scalable storage solution"  

vector_query = VectorizableTextQuery(text=query, k_nearest_neighbors=50, fields="contentVector", weight=0.2)

results = search_client.search(  
    search_text=query,  
    vector_queries=[vector_query],
    select=["title", "content", "category"],
    top=3
)  

print_results(results)


# ## Perform a Semantic Hybrid Search

# In[26]:


from azure.search.documents.models import QueryType, QueryCaptionType, QueryAnswerType

# Semantic Hybrid Search
query = "what is azure sarch?"

vector_query = VectorizableTextQuery(text=query, k_nearest_neighbors=50, fields="contentVector", exhaustive=True)

results = search_client.search(  
    search_text=query,  
    vector_queries=[vector_query],
    select=["title", "content", "category"],
    query_type=QueryType.SEMANTIC,
    semantic_configuration_name='my-semantic-config',
    query_caption=QueryCaptionType.EXTRACTIVE,
    query_answer=QueryAnswerType.EXTRACTIVE,
    top=3
)

print_results(results)


# ## Perform a Semantic Hybrid Search with Query Rewriting
# 
# + Uses [query rewriting](https://learn.microsoft.com/azure/search/semantic-how-to-query-rewrite) combined with semantic ranker
# + Set the debug parameter to queryRewrites to see what the query was rewritten to
# + Avoid using the debug parameter in production

# In[27]:


from azure.search.documents.models import QueryType, QueryCaptionType, QueryAnswerType, QueryDebugMode
from typing import Optional

# Workaround required to use debug query rewrites with the preview SDK
import azure.search.documents._generated.models
azure.search.documents._generated.models.SearchDocumentsResult._attribute_map["debug_info"]["key"] = "@search\\.debug"
from azure.search.documents._generated.models import DebugInfo
import azure.search.documents._paging
def get_debug_info(self) -> Optional[DebugInfo]:
    self.continuation_token = None
    return self._response.debug_info
azure.search.documents._paging.SearchPageIterator.get_debug_info = azure.search.documents._paging._ensure_response(get_debug_info)
azure.search.documents._paging.SearchItemPaged.get_debug_info = lambda self: self._first_iterator_instance().get_debug_info()

# Semantic Hybrid Search with Query Rewriting
query = "what is azure sarch?"

vector_query = VectorizableTextQuery(text=query, k_nearest_neighbors=50, fields="contentVector", query_rewrites="generative|count-5")

results = search_client.search(  
    search_text=query,  
    vector_queries=[vector_query],
    select=["title", "content", "category"],
    query_type=QueryType.SEMANTIC,
    semantic_configuration_name='my-semantic-config',
    query_rewrites="generative|count-5",
    query_language="en",
    debug=QueryDebugMode.QUERY_REWRITES,
    query_caption=QueryCaptionType.EXTRACTIVE,
    query_answer=QueryAnswerType.EXTRACTIVE,
    top=3
)

text_query_rewrites = results.get_debug_info().query_rewrites.text.rewrites
vector_query_rewrites = results.get_debug_info().query_rewrites.vectors[0].rewrites
print_results(results)

print("Text Query Rewrites:")
print(text_query_rewrites)
print("Vector Query Rewrites:")
print(vector_query_rewrites)

