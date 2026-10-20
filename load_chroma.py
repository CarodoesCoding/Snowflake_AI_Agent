from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
import hashlib


def load_chroma_db(loc, col_name, model):
    """lädt die Chroma Vektordatenbank mit Ollama-Embeddings, um sie für das RAG zu verwenden"""
    embeddings = OllamaEmbeddings(model=model)
    vectordb = Chroma(
        collection_name=col_name, persist_directory=loc, embedding_function=embeddings
    )
    return vectordb


def rag(query, vectordb, k):
    """durchsucht die Chroma Vektordatenbank nach ähnlichen Fragen"""
    most_similar = vectordb.similarity_search(query, k=k)
    result = []
    for doc in most_similar:
        content = doc.page_content
        try:
            question = content.split("Frage: ")[1].split(" SQL: ")[0].strip()
            sql = content.split(" SQL: ")[1].strip()
        except IndexError:
            question, sql = content, ""
        result.append({"question": question, "sql": sql})
    return result


def add_pos_example(vectordb, question: str, sql: str):
    """speichert Frage und SQL im RAG System, wenn von Nutzer im Dashboard gut bewertet"""
    if not question or not sql:
        return

    page_content = f"Frage: {question} SQL: {sql}"

    example_id = hashlib.md5(page_content.encode("utf-8")).hexdigest()

    doc = Document(
        page_content=page_content,
        metadata={
            "source": "user_positive_rating",
            "question": question,
        },
    )

    vectordb.add_documents(
        documents=[doc],
        ids=[example_id],
    )


# test
if __name__ == "__main__":
    db = load_chroma_db("./chroma_db", "business_questions", "snowflake-arctic-embed2")
    most_similar = rag("Wie viele hörer gab es diese Woche?", db, 2)
    print(most_similar)
