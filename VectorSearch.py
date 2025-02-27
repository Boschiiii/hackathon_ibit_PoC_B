import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings

if load_dotenv():
    print("Env-Datei wurde erfolgreich geladen.")
else:
    print("Keine Env-Datei gefunden.")

azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

print(azure_deployment)
print(api_key)
print(azure_endpoint)

azure_openai_embeddings = AzureOpenAIEmbeddings(
    azure_endpoint = azure_endpoint,
    azure_deployment= azure_deployment,
    api_key = api_key
)
print("Embedding-Client mit Azure OpenAI initialisiert.")

user_query = "Mein drucker hat kein papier mehr"

# Embedding erstellen
query_vector = azure_openai_embeddings.embed_query(user_query)
vector_query = VectorizedQuery(
    vector=query_vector,
    k_nearest_neighbors=5,  # Anzahl der ähnlichen Treffer
    fields="vector"  # Das Feld im Index, das Embeddings enthält
)

# Alle durchsuchbaren Felder (entsprechend den Feldern in deinem Index)
searchable_fields = [
    "original_language", "original_title", "popularity", "release_date", 
    "vote_average", "vote_count", "genre", "overview", "revenue", 
    "runtime", "tagline"
]

# Suchanfrage an Azure AI Search
search_results = search_client.search(
    search_text=None,  # Reine Vektorsuche, kein Text
    search_fields=searchable_fields,  # Suche in allen relevanten Feldern
    vector_queries=[vector_query],  # Vektorsuche aktiv
    top=5
)

results_list = list(search_results)
for result in results_list:
    print(f"Titel: {result['original_title']} | Genre: {result.get('genre', 'keine Angabe')} | Score: {result['@search.score']}")