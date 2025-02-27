import streamlit as st
import os
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from langchain.callbacks.base import BaseCallbackHandler
from azure.search.documents.models import VectorizedQuery
from flask import Flask, request, jsonify


# Lade Umgebungsvariablen
load_dotenv()

# Azure AI Search & CosmosDB Config
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_AI_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_AI_SEARCH_API_KEY")
AZURE_SEARCH_INDEX = os.getenv("AZURE_AI_SEARCH_INDEX_NAME")

# OpenAI Config
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT_NAME")
EMBEDDING_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")


# Set up Hybrid Search with Azure AI Search
def setup_azure_ai():
    """Initialisiert Azure AI Search für Hybrid-Suche."""
    
    # Embedding Model für Vektor-Suche
    embeddings_model = AzureOpenAIEmbeddings(
        azure_deployment=EMBEDDING_NAME,
        chunk_size=1000
    )

    # Azure AI Search Client
    search_client = SearchClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        index_name=AZURE_SEARCH_INDEX,
        credential=AzureKeyCredential(AZURE_SEARCH_KEY)
    )

    # LLM (GPT) Initialisierung
    llm = AzureChatOpenAI(
        azure_deployment = os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT_NAME")
    )

    return search_client, embeddings_model, llm


# Azure AI Hybrid Search
def hybrid_search(query, search_client, embeddings_model):
    """Führt eine Hybrid-Suche mit Azure AI Search durch (Keyword + Vector + Semantic)."""
    
    # Vektor-Erstellung für semantische Suche 
    query_vector = embeddings_model.embed_query(query)

    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=5,  # Anzahl der ähnlichen Treffer
        fields="vector"  # Das Feld im Index, das Embeddings enthält
    )

    # Azure AI Search Abfrage 
    search_results = search_client.search(
        search_text=query,  # Textbasierte Suche
        vector_queries=[vector_query],  # Vektorbasierte Suche
        query_type="semantic",  # Semantische Suche aktivieren
        semantic_configuration_name="ibit-semantic-config",
        select=["KBAName","title","problem","solution"],
        top=5
    )

    #  Ergebnisse sortieren nach kombiniertem Search Score
    sorted_results = sorted(list(search_results), key=lambda x: x.get('@search.score', 0), reverse=True)


    #  Ergebnisse formatieren 
    context = []
    for item in sorted_results:
        context.append(
            f"  -  📊 Search Score: {item.get('@search.score', 'N/A')}\n"
            f"  -  🎬 KBA-ID: {item.get('KBAName', 'N/A')}\n"
            f"  -  🌍 Titel: {item.get('title', 'N/A')}\n"
            f"  -  📅 Frage: {item.get('problem', 'N/A')}\n"
            f"  -  ⭐ Antwort: {item.get('solution', 'N/A')} ({item.get('vote_count', 'N/A')} Stimmen)\n"
        )

    return "\n".join(context)



#  Streamlit Callback Handlers 
class StreamHandler(BaseCallbackHandler):
    """Verarbeitet LLM-Streaming-Ausgabe für die UI."""
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str,  kwargs):
        self.text += token
        self.container.info(self.text)


#  Streamlit UI Setup 
st.sidebar.image("it-support.png")
st.header("`KBA Agent`")
st.info("Ich bin ein KI-Agent, der die KBA's suchen kann.")

# Initialisierung von Azure AI Search & OpenAI Embeddings
if 'search_client' not in st.session_state:
    st.session_state['search_client'], st.session_state['embeddings_model'], st.session_state['llm'] = setup_azure_ai()

search_client = st.session_state.search_client
embeddings_model = st.session_state.embeddings_model
llm = st.session_state.llm

#  Benutzerabfrage 
question = st.text_input("🎤 `Stell deine IT-Frage`")

if question:
    #  Hybrid AI Search starten 
    film_kontext = hybrid_search(question, search_client, embeddings_model)

    #  Prompt Template für LLM 
    prompt_template = PromptTemplate(
        input_variables=["frage", "film_kontext"],
        template="""Du bist IBIT, ein KI-gestützter Assistent für IT-Supporter im 1st-Level-Support. Deine Aufgabe ist es, Supportern bei der schnellen und effizienten Suche nach passenden KBAs zu helfen.  
        Frage: {frage}
        Hier sind die relevanten Informationen, die als Kontext dienen:
        {film_kontext}
        \n**So sollst du antworten:**  
        \n✅ **Kurz & präzise:** Antworte in 3-5 Sätzen mit dem Kern der Lösung.  
        \n✅ **KBA-Suche optimieren:** Nutze Titel, Fragen, Beschreibung und Lösungsabschnitt der KBAs zur Identifikation der besten Treffer. Falls du unsicher bist, nenne bis zu 3 wahrscheinlichste KBAs mit Wahrscheinlichkeitswerten.  
        \n✅ **Falls nichts gefunden:** Schlage allgemeine Troubleshooting-Schritte vor.  
        \n✅ **Tonalität:** Antworte in der Du-Form, fachlich-locker, freundlich und klar. Sei unterstützend, um den Supporter in seinem stressigen Alltag zu entlasten.  
        \n✅ **Zusatzfunktionen:** Falls eine KBA bestätigt wurde, biete Optionen zur Anpassung der Anrede (Du/Sie) und zur Übersetzung (Deutsch, Englisch, Französisch, Italienisch) an.  
        \nBeispielantwort:  
        \n📌 *„Die KBA [Titel] beschreibt dein Problem exakt. Hier ist die Lösung in Kürze: … Falls das nicht passt, könnten diese Alternativen helfen: [KBA1] (85%), [KBA2] (72%).*”"
        """
    )

    #  Pipeline definieren 
    chain = LLMChain(llm=llm, prompt=prompt_template)

    #  Antwort generieren 
    answer = st.empty()
    stream_handler = StreamHandler(answer, initial_text="`💡 Antwort:`\n\n")
    
    result = chain.run({"frage": question, "film_kontext": film_kontext}, callbacks=[stream_handler])

    #  Ausgabe formatieren 
    st.subheader("🎬 `Antwort:`")
    st.info(result)
    
    st.subheader("📚 `Relevante KBA's:`")
    st.markdown(film_kontext)