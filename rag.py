from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Load a document about AI
loader = WebBaseLoader("https://python.langchain.com/v0.2/docs/introduction/")
documents = loader.load()

# 2. Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", ". ", " ", ""])
chunks = text_splitter.split_documents(documents)

# 3. Set up the embedding model (free local model, no API key required).
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. Create a vector store
vector_store = Chroma.from_documents(chunks, embedding_model)

# 5. Create a retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# 6. Define a function to search for relevant information
def search_documents(query, top_k=3):
    """Search for documents relevant to a query"""
    # Use the retriever to get relevant documents
    docs = retriever.invoke(query)
    
    # Limit to top_k if specified
    return docs[:top_k]

# 7. Test with a few queries
test_queries = [
    "What is LangChain?",
    "How do retrievers work?",
    "Why is document splitting important?"
]

for query in test_queries:
    print(f"\nQuery: {query}")
    results = search_documents(query)
    # Print the results
    print(f"Found {len(results)} relevant documents:")
    for i, doc in enumerate(results):
        print(f"\nResult {i+1}: {doc.page_content[:150]}...")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")