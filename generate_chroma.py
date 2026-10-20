from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_chroma import Chroma

test_dict = [
    {
        "Natural Language": "EXAMPLE QUESTION",
        "SQL": "EXAMPLE SQL REQUEST"
    },
]


example_dict = [
    {
        "Natural Language": "EXAMPLE QUESTION",
        "SQL": "EXAMPLE SQL REQUEST"
    },
]


def generate_chroma_db(dictionary, loc, col_name,  model):
    documents = [Document(page_content=f"Frage: {d['Natural Language']} SQL: {d['SQL']}") for d in dictionary]
    embeddings = OllamaEmbeddings(model=model)
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=col_name,
        persist_directory=loc
    )

generate_chroma_db(example_dict, "./chroma_db", "business_questions", "snowflake-arctic-embed2")
