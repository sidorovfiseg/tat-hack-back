from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import json
from time import perf_counter

import chromadb
from chromadb import EmbeddingFunction, Embeddings, Documents
from fastapi import APIRouter
from langchain.retrievers import ContextualCompressionRetriever
from langchain_chroma import Chroma
from langchain_community.chat_models.gigachat import GigaChat
from langchain_community.embeddings import GigaChatEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_voyageai import VoyageAIRerank
from pydantic import BaseModel

import pprint 
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


class GigaChatApiConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='GIGA_CHAT_API_')

    TOKEN: str = Field('NmNhYjhmZGEtNmFmNi00MDAwLTg5NWMtNWRiZDJjYjRjN2E1OjUxMzM1Yzk1LWUzY2EtNDJjZC1iMTRjLWJlMzllM2JjYTYyYQ==')
    SCOPE: Optional[str] = Field('GIGACHAT_API_CORP')


giga_chat_api_config = GigaChatApiConfig()



class GigaChatEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(self, credentials: str, scope: str):
        self.embeddings = GigaChatEmbeddings(credentials=credentials, verify_ssl_certs=False, scope=scope)

    def __call__(self, input: Documents) -> Embeddings:
        return self.embeddings.embed_documents(texts=input)

    def embed_query(self, query: str) -> Embeddings:
        print(f'{query=}')
        return self.embeddings.embed_documents([query])
    
gigachat_embedding = GigaChatEmbeddings(credentials=giga_chat_api_config.TOKEN, scope=giga_chat_api_config.SCOPE, verify_ssl_certs=False)
gigachat_embedding_function = GigaChatEmbeddingFunction(credentials=giga_chat_api_config.TOKEN, scope=giga_chat_api_config.SCOPE)
gigachat_llm = GigaChat(credentials=giga_chat_api_config.TOKEN, scope=giga_chat_api_config.SCOPE, verify_ssl_certs=False, model='GigaChat-Plus', profanity_check=None, verbose=False)

chroma_client = chromadb.HttpClient(host="localhost", port=8000)
gigachat_tink_collection = chroma_client.get_or_create_collection(name = "gigachat_tink_collection", embedding_function=gigachat_embedding_function)
gigachat_tink_collection_retriever = Chroma( client = chroma_client, collection_name = gigachat_tink_collection.name, embedding_function=gigachat_embedding_function.embeddings,).as_retriever(search_kwargs={'k': 10})

import pandas as pd
from time import perf_counter

# Шаг 1: Загрузка и обработка данных
df = pd.read_excel(r"тестовый_датасет_с_самамри.xlsx")

# Печать количества строк до и после удаления пустых ID
print(len(df))
df = df.dropna(subset=[df.columns[5]])
print(len(df))

# Приведение значений ID к строковому типу
df.iloc[:, 5] = df.iloc[:, 5].astype(str)

# Найдем и выведем дублирующиеся ID
duplicate_ids = df[df.duplicated(subset=[df.columns[5]], keep=False)]
print("Дублирующиеся ID и их данные:")
print(duplicate_ids)

# Удаляем дубликаты, оставляя только одну запись для каждого уникального ID
df = df.drop_duplicates(subset=[df.columns[5]])

# Обрезаем текст в 5-м столбце до 250 символов и удаляем слова "Введение:" и "Заголовок:"
def clean_text(text):
    """Функция для очистки текста."""
    text = str(text)
    # Удаляем слова "Введение:" и "Заголовок:"
    #text = text.replace("Введение:", "").replace("Заголовок:", "")
    # Обрезаем текст до 250 символов
    return text[:250]

df.iloc[:, 4] = df.iloc[:, 4].apply(clean_text)

# Получаем данные из столбцов
documents = df.iloc[:, 4].tolist()  # Обрезанное содержимое 5-го столбца
ids = df.iloc[:, 5].tolist()  # ID из 6-го столбца
summaries = df.iloc[:, 4].tolist()  # Саммари из 4-го столбца (предположительно)

# Преобразуем их в формат документов с метаданными, включая саммари
documents_with_metadata = [
    {
        "page_content": doc,
        "metadata": {
            "id": id_,
            "summary": summary  # Добавляем саммари в метаданные
        }
    }
    for doc, id_, summary in zip(documents, ids, summaries)
]

# Шаг 2: Функция для разбивки данных на батчи
def chunkify(data, chunk_size):
    """Функция для разбиения данных на пакеты."""
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]

# Шаг 3: Обновленная функция embed_documents_to_collection
def embed_documents_to_collection(collection, documents, embedding_function, chunk_size=10):
    """Добавляет документы в коллекцию, разбивая их на пакеты."""
    start = perf_counter()

    # Разбиваем документы на пакеты и обрабатываем их
    for chunk in chunkify(documents, chunk_size):
        # Подготовка данных для текущего пакета
        ids_new = [doc['metadata']['id'] for doc in chunk]
        documents_new = [doc['page_content'] for doc in chunk]
        metadatas_new = [doc['metadata'] for doc in chunk]

        # Проверяем, есть ли эти документы уже в коллекции
        existing_documents = collection.get(ids=ids_new, include=["metadatas"])["metadatas"]
        existing_ids = {metadata['id'] for metadata in existing_documents}
        new_documents = [doc for doc in chunk if doc['metadata']['id'] not in existing_ids]

        if not new_documents:
            continue

        # Вычисляем эмбеддинги для текущего пакета
        embeddings = embedding_function(documents_new)

        # Добавляем новые документы в коллекцию
        collection.add(ids=ids_new, documents=documents_new, metadatas=metadatas_new, embeddings=embeddings)

        print(f'{collection.name} добавлено документов: {len(new_documents)}, время выполнения: {perf_counter() - start:.2f} секунд')

# Шаг 4: Вызов функции embed_documents_to_collection с батчевой обработкой
embed_documents_to_collection(gigachat_tink_collection, documents_with_metadata, gigachat_embedding_function.embeddings)
app = FastAPI()
class ChatRequest(BaseModel):
    text: str

prompt = PromptTemplate.from_template("""Ты - русскоязычный ассистент для ответа на вопросы попатентам. Отвечай только на вопрос основываясь только на следующих контекстах.
Вопрос: {question}
Контексты: {context}
Ответ:""")


def format_summaries(docs):
    """Функция для форматирования саммари из метаданных документов."""
    return "\n\n".join(doc.metadata["summary"] for doc in docs if "summary" in doc.metadata)


@app.post("/chat/")
async def chat_endpoint(message: ChatRequest):
    try:
        response = await chat(message)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Функции, которые вы уже определили, можно использовать без изменений
# Например:
async def chat(
        message: ChatRequest
):
    # Обновляем цепочку, чтобы использовать только саммари
    rag_chain_from_summaries = (
            RunnablePassthrough.assign(context=(lambda x: format_summaries(x["context"])))
            | prompt
            | gigachat_llm
            | StrOutputParser()
    )

    # Обновляем цепочку для поиска контекста
    rag_chain_with_source = RunnableParallel(
        {"context": gigachat_tink_collection_retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_summaries)

    # Запрос к модели с использованием только саммари в качестве контекста
    response = await rag_chain_with_source.ainvoke(message.text)
    return response


# Для тестирования FastAPI-приложения
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)