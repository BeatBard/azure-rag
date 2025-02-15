import os
import faiss
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document
import uvicorn

# Import openai and check version
import openai
print("OpenAI SDK version:", openai.__version__)

# Use the new OpenAI client interface (requires openai>=1.0.0)
try:
    from openai import OpenAI
    client_openai = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", "sk-no-key-required"),
        base_url="http://127.0.0.1:8080/v1"
    )
except ImportError as e:
    raise ImportError("Failed to import OpenAI from the openai package. Ensure you have upgraded to openai>=1.0.0.") from e

# ---------------------------
# Create FastAPI app instance BEFORE using it!
app = FastAPI()

# File paths
CSV_PATH = "../wine-ratings.csv"  # Path to the CSV dataset
DB_PATH = "faiss_index"           # Directory to store the FAISS index

# Initialize the embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embedding_dim = len(embeddings.embed_query("test"))  # Compute dimension dynamically
print(f"✅ Embedding dimension detected as: {embedding_dim}")

# Load or create the FAISS vector store with the correct dimension
if os.path.exists(DB_PATH):
    print("📌 Loading existing FAISS index...")
    vector_store = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
    print(f"📌 FAISS index contains {vector_store.index.ntotal} vectors.")

    # ✅ DEBUG: Print sample documents to check if FAISS is storing full details
    print("📄 Sample stored documents from FAISS:")
    sample_docs = vector_store.similarity_search("random", k=3)
    for i, doc in enumerate(sample_docs):
        print(f"📜 Document {i+1}: {doc.page_content[:200]}")  # Print first 200 characters

else:
    print("📌 Creating a new FAISS index...")
    vector_store = FAISS(
        embeddings,
        faiss.IndexFlatL2(embedding_dim),
        InMemoryDocstore({}),
        index_to_docstore_id={}
    )
    
    df = pd.read_csv(CSV_PATH)
    print(f"✅ CSV loaded with {len(df)} rows.")
    
    # ✅ Optimized FAISS Indexing (Uses df.to_dict('records') for faster processing)
    records = df.to_dict('records')

    docs = [
        Document(
            page_content=(
                f"Name: {row.get('name', 'Unknown Wine')}. "
                f"Grape: {row.get('grape', 'Unknown Grape')}. "
                f"Region: {row.get('region', 'Unknown Region')}. "
                f"Variety: {row.get('variety', 'Unknown Variety')}. "
                f"Rating: {row.get('rating', 'No Rating')}. "
                f"Notes: {row.get('notes', 'No Notes Available')}."
            )
        )
        for row in records
    ]

    print(f"✅ FAISS will be populated with {len(docs)} rich-text wine documents.")
    vector_store.add_documents(docs)
    vector_store.save_local(DB_PATH)
    print(f"✅ FAISS index successfully rebuilt with {len(docs)} documents!")

# Define the request body model for FastAPI
class Body(BaseModel):
    query: str

@app.get('/')
def root():
    return RedirectResponse(url='/docs', status_code=301)

@app.post('/ask')
def ask(body: Body):
    print("🔍 Received query:", body.query)
    search_result = search(body.query)
    print("🔍 Search completed. Result (first 200 chars):", search_result[:200])
    
    chat_bot_response = assistant(body.query, search_result)
    print("🗣️ Assistant response (first 200 chars):", chat_bot_response[:200])
    
    return {'response': chat_bot_response}

def search(query):
    docs = vector_store.similarity_search(query, k=5)
    results = " ".join([doc.page_content for doc in docs])
    return results

def assistant(query, context):
    # ✅ Only send the **top search result** to prevent API overload
    top_result = context.split(". ")[0] if context else "No relevant data found."

    # ✅ Structured prompt using the user’s actual query
    messages = [
        {"role": "system", "content": "You are chatbot, a wine specialist. Your top priority is to help guide users into selecting amazing wine and guide them with their requests."},
        {"role": "user", "content": query},  # ✅ Uses the actual user input
        {"role": "assistant", "content": f"Here is the wine information I found: {top_result}"},
        {"role": "user", "content": "Based on this, what makes this wine special?"}  # ✅ Makes the model generate a useful response
    ]

    try:
        response = client_openai.chat.completions.create(
            model="LLaMA_CPP",
            messages=messages,
            max_tokens=100,  # ✅ Faster response
            timeout=150,
            temperature=0.7  # ✅ Balanced creativity
        )

        print("🔹 Llamafile API Response:", response)
        
        if hasattr(response, "choices") and len(response.choices) > 0:
            return response.choices[0].message.content
        else:
            return "⚠️ Error: Unexpected response format from LLaMA_CPP."

    except Exception as e:
        print(f"❌ Error calling LLaMA_CPP: {e}")
        return f"⚠️ Error: {str(e)}"



@app.on_event("shutdown")
def save_faiss_index():
    vector_store.save_local(DB_PATH)
    print("📌 FAISS index saved successfully.")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
