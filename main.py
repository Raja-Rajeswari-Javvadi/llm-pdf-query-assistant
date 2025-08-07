import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

# -------- Step 1: Read PDF --------
def read_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

# -------- Step 2: Split into Chunks --------
def chunk_text(text, max_words=200):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i+max_words])
        chunks.append(chunk)
    return chunks

# -------- Step 3: Create Embeddings --------
def create_embeddings(chunks, model):
    embeddings = model.encode(chunks)
    return np.array(embeddings)

# -------- Step 4: Create FAISS Index --------
def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# -------- Step 5: Search Index --------
def search_index(index, chunks, query, model, top_k=2):
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), top_k)
    results = [(D[0][i], chunks[I[0][i]]) for i in range(top_k)]
    print("\n🔎 Top Matching Chunks:\n")
    for score, chunk in results:
        print(f"Result (Score: {score:.4f}):\n{chunk}\n")
    return results

# -------- Step 6: Use LLM to Answer --------
def answer_query_with_llm(query, context):
    qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

    prompt = f"""
    You are a smart insurance policy assistant. Based on the policy text below, answer the question. 
    Respond in this JSON format: 
    {{
        "eligibility": "Yes/No",
        "reasoning": "Detailed explanation based only on policy terms"
    }}

    Policy Document:
    \"\"\"{context}\"\"\"

    Question:
    \"\"\"{query}\"\"\"
    """

    response = qa_pipeline(prompt, max_length=512, do_sample=False)[0]['generated_text']
    print("\n🤖 Final Answer:\n", response)

# -------- Main --------
if __name__ == "__main__":
    file_path = "sample_policy.pdf"  # Make sure this PDF exists in the same folder
    text = read_pdf(file_path)
    chunks = chunk_text(text)

    print(f"✅ Total Chunks: {len(chunks)}\n")
    print("🧾 Sample Chunk:\n", chunks[0])

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = create_embeddings(chunks, model)
    print(f"\n✅ Embeddings Created — Shape: {embeddings.shape}")

    index = create_faiss_index(embeddings)

    query = input("\n🔍 Enter your query: ")
    results = search_index(index, chunks, query, model)

    if results:
        top_context = results[0][1]
        answer_query_with_llm(query, top_context)
    else:
        print("❌ No relevant information found.")
