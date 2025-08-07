import os
import torch
import gradio as gr
import PyPDF2
from sentence_transformers import SentenceTransformer
from InstructorEmbedding import INSTRUCTOR
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity

# Load models once
embedding_model = INSTRUCTOR('hkunlp/instructor-base')
qa_model = pipeline("text2text-generation", model="google/flan-t5-base")

# Helper: Extract and chunk text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()

def chunk_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# In-memory store
document_chunks = []
document_embeddings = []

def compute_embeddings(chunks):
    instruction = "Represent the document chunk for retrieval:"
    return embedding_model.encode([[instruction, chunk] for chunk in chunks])

# PDF Upload Logic
def upload_pdf(pdf_file):
    global document_chunks, document_embeddings
    text = extract_text_from_pdf(pdf_file.name)
    document_chunks = chunk_text(text)
    document_embeddings = compute_embeddings(document_chunks)
    return f"✅ PDF uploaded successfully! {len(document_chunks)} chunks created."

# Question Answering
def answer_query(query):
    if not document_chunks:
        return "❌ Please upload a PDF first."
    
    instruction = "Represent the question for retrieving supporting documents:"
    query_embedding = embedding_model.encode([[instruction, query]])

    similarities = cosine_similarity(query_embedding, document_embeddings)[0]
    top_indices = similarities.argsort()[-3:][::-1]  # Top 3
    top_chunks = "\n\n".join([document_chunks[i] for i in top_indices])

    prompt = f"Answer the question based on the document:\n\n{top_chunks}\n\nQuestion: {query}\nAnswer:"

    result = qa_model(prompt, max_length=256, do_sample=False)[0]["generated_text"]

    return f"📄 **Top Matching Chunk:**\n{top_chunk[:500]}...\n\n🤖 **Answer:**\n{result}"

# Build UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🤖 LLM-Powered PDF Query Assistant")
    gr.Markdown(
        "Upload your PDF 📄 and ask questions in natural language 💬. "
        "Get instant answers from document content using LLMs!"
    )

    with gr.Group():
        with gr.Row():
            pdf_input = gr.File(label="📂 Upload PDF File", file_types=[".pdf"])
            upload_button = gr.Button("📤 Process PDF", variant="primary")
        upload_status = gr.Textbox(label="Status", interactive=False)

    with gr.Group():
        gr.Markdown("### 🔍 Ask a Question")
        query_input = gr.Textbox(label="Type your question here...", placeholder="e.g. What is the coverage for knee surgery?")
        query_button = gr.Button("💬 Get Answer", variant="secondary")
        query_output = gr.Textbox(label="Answer", lines=10, interactive=False)

    upload_button.click(upload_pdf, inputs=[pdf_input], outputs=[upload_status])
    query_button.click(answer_query, inputs=[query_input], outputs=[query_output])

# Launch app
demo.launch()
