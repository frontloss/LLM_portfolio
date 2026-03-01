import streamlit as st
import yaml, os
from typing import Any

from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import (
    StorageContext, 
    load_index_from_storage, 
    Settings, 
    PromptTemplate 
)
from llama_index.core.base.base_query_engine import BaseQueryEngine
from dotenv import load_dotenv
load_dotenv()

chat_llm = AzureOpenAI(
    engine=os.getenv('AZURE_OPENAI_CHAT_DEPLOYMENT_NAME'),
    model = "gpt-4o",
    api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT')
)

embedding_llm = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

Settings.llm = chat_llm
Settings.embed_model = embedding_llm

@st.cache_resource(show_spinner=False)
def load_index() -> BaseQueryEngine:
    """Load the index and configure the system prompt."""
    print("Loading index...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dir_path = os.path.join(base_dir, "kb")
    storage_context = StorageContext.from_defaults(persist_dir=dir_path)
    index = load_index_from_storage(storage_context)
    system_prompt = (
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Your purpose is to answer questions about specific documents only. "
        "Please answer the user's questions based on the provided context. "
        "If the question is outside the scope of the document, politely decline. "
        "If you don't know the answer, say 'I don't know'.\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    qa_prompt_tmpl = PromptTemplate(system_prompt)
    
    query_engine = index.as_query_engine(similarity_top_k=3)
    query_engine.update_prompts(
        {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
    )
    
    print("Done.")
    return query_engine

def main() -> None:
    st.title("Chat with BlogAI Assistant!!")
    st.write("Ask questions about Snowpark for Data Engineering.")
    if "query_engine" not in st.session_state:
        st.session_state.query_engine = load_index()
        
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = st.session_state.query_engine.query(prompt)
            full_response = str(response)
            st.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()