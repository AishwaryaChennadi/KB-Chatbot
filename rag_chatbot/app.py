import os
import requests
import gradio as gr
from dotenv import load_dotenv
from pathlib import Path
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# -----------------------------
# 1. Load environment variables
# -----------------------------
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# -----------------------------
# 2. Define DeepSeek LLM wrapper
# -----------------------------
class DeepSeekLLM:
    def __init__(self, model="deepseek/deepseek-chat"):
        self.model = model

    def invoke(self, prompt):
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}]
        }
        resp = requests.post(url, headers=headers, json=payload)

        try:
            data = resp.json()
        except Exception:
            return f"API Error: Could not parse response. Raw: {resp.text}"

        if "error" in data:
            return f"API Error: {data['error'].get('message', 'Unknown error')}"
        if "choices" not in data:
            return f"Unexpected API response: {data}"
        return data["choices"][0]["message"]["content"]

# -----------------------------
# 3. Setup ChromaDB with private data
# -----------------------------
docs = []
data_path = Path("data/mydata.txt")
if data_path.exists():
    loader = TextLoader(str(data_path))
    docs.extend(loader.load())
    print(f"✅ Loaded {len(docs)} documents from {data_path}")
else:
    print("⚠️ mydata.txt not found in ./data/")

splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50)
chunks = splitter.split_documents(docs)
print(f"✅ Created {len(chunks)} chunks")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectordb = None
retriever = None
if chunks:
    try:
        vectordb = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
        #vectordb.persist()
        retriever = Chroma(persist_directory="./chroma_db", embedding_function=embeddings).as_retriever()
        print("✅ Vector DB created successfully")
    except Exception as e:
        print(f"❌ Error creating vector DB: {e}")
else:
    print("⚠️ No chunks available, skipping vector DB creation")

llm = DeepSeekLLM()

# -----------------------------
# 4. RAG query function
# -----------------------------
def rag_query(query):
    if retriever is None:
        return "⚠️ No knowledge base available. Please add data/mydata.txt and restart."
    
    docs = retriever.invoke(query)   # safer than invoke
    print(f"Retrieved {len(docs)} docs")
    
    if docs:
        context = "\n".join([d.page_content for d in docs])
        prompt = f"Use the following context to answer:\n{context}\n\nQuestion: {query}"
    else:
        prompt = f"Answer the question directly: {query}"
    return llm.invoke(prompt)

# -----------------------------
# 5. Gradio UI
# -----------------------------
def chat(message, history):
    return rag_query(message)

gr.ChatInterface(
    fn=chat,
    chatbot=gr.Chatbot(label="ThinkLet"),
    title="ThinkLet Chatbot",
    description="Hi, I am ThinkLet! How can I help you today?"
).launch(server_port=7861)
