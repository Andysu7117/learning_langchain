# Importing necessary modules from the chromadb package:
# chromadb is used to interact with the Chroma DB database,
# embedding_functions is used to define the embedding model
import chromadb
from chromadb.utils import embedding_functions

# Define the embedding function using SentenceTransformers
# This function will be used to generate embeddings (vector representations) for the data
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Creating an instance of ChromaClient to establish a connection with the Chroma database
client = chromadb.Client()

# Defining a name for the collection where data will be stored or accessed
# This collection is likely used to group related records, such as employee data
collection_name = "employee_collection"

# Defining a function named 'main'
# This function is used to encapsulate the main operations for creating collections,
# generating embeddings, and performing similarity search
def main():
    try:
# Creating a collection using the ChromaClient instance
# The 'create_collection' method creates a new collection with the specified configuration
        collection = client.create_collection(
            # Specifying the name of the collection to be created
            name=collection_name,
            # Adding metadata to describe the collection
            metadata={"description": "A collection for storing employee data"},
            # Configuring the collection with cosine distance and embedding function
            configuration={
                "hnsw": {"space": "cosine"},
                "embedding_function": ef
            }
        )
        print(f"Collection created: {collection.name}")
    except Exception as error:
        # Catching and handling any errors that occur within the 'try' block
        # Logs the error message to the console for debugging purposes
        print(f"Error: {error}")