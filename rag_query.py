from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama

"""
Loads the existing FAISS policy database, retrieves the 3 most relevant
passages for a question, and asks Llama 3.2 to answer using only those passages.

"""

# Load the same embedding model used to build the FAISS index.
# This is where converts your policy text and user questions into numerical representations
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")  # must match the build step

# Load the saved policy vector database.
db = FAISS.load_local("./rag_faiss_index", embeddings, allow_dangerous_deserialization=True)

# Initialize the LLM
llm = ChatOllama(model="qwen2.5:7b", temperature=0)

# Define the question to ask the policy documents.
query = "Can AP write off a supplier credit balance without approval if the amount is small?"

# Retrieve the 3 most relevant policy passages.
docs = db.similarity_search(query, k=3)

# Print the first 200 characters of each retrieved document
for doc in docs:
    print(doc.page_content[:200].replace("\n", " "))
    print("---")

# Combine retrieved passages into context for the model.
context = "\n\n".join(d.page_content for d in docs)

# Instruct the model to answer using only the retrieved policy excerpts.
prompt = f"""Answer the question using only the policy excerpts below.

Rules:
- Prefer explicit policy names and policy reference numbers when asked which policy governs a requirement.
- Do not confuse a section heading with the name of a governing policy.
- Do not invent or change policy titles or reference numbers.
- If the answer is not explicitly supported by the excerpts, say you don't know.

Excerpts:
{context}

Question: {query}

Answer:"""

print(llm.invoke(prompt).content)