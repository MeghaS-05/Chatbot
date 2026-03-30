import os

# LangChain modules
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import GPT4All
from langchain.chains import RetrievalQA

# Step 1: Ask user for folder path
pdf_folder = input("Enter the path to your PDF folder: ").strip()

if not os.path.exists(pdf_folder):
    print("Folder not found. Please check the path.")
    exit()

# Step 2: Load PDFs
pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
if not pdf_files:
    print("No PDF files found in that folder.")
    exit()

print(f"Found {len(pdf_files)} PDF(s). Loading...")

docs = []
for f in pdf_files:
    loader = PyPDFLoader(os.path.join(pdf_folder, f))
    docs.extend(loader.load())
    print(f"Loaded: {f}")

# Step 3: Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# Step 4: Create embeddings and vector store
embedding = GPT4AllEmbeddings()
db = Chroma.from_documents(chunks, embedding)

# Step 5: Load Mistral model
llm = GPT4All(
    model="mistral-instruct-v0.1.Q4_0.bin",
    model_path="C:/Users/lenovo/AppData/Local/nomic.ai/GPT4All/",
    verbose=False
)

# Step 6: Create QA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

print("PDFs indexed. You can now ask questions.")
print("Type your question below. Type 'exit' to quit.\n")

# Step 7: Chat loop
while True:
    query = input("You: ").strip()
    if query.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break
    try:
        answer = qa_chain.run(query)
        print(f"Bot: {answer}\n")
    except Exception as e:
        print(f"Error: {e}\n")