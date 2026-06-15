import chromadb
from chromadb.utils import embedding_functions

client = chromadb.Client()

collection_name = "my_grocery_collection"

def main():
    try:
        collection = client.create_collection(
            name=collection_name,
            metadata={"description": "A collection for storing grocery data"},
            configuration={
                "hnsw": {"space": "cosine"},
                "embedding_function": ef
            }
        )
        print(f"Collection created: {collection.name}")
    except Exception as error:
        print(f"Error: {error}")

texts = [
    'fresh red apples',
    'organic bananas',
    'ripe mangoes',
    'whole wheat bread',
    'farm-fresh eggs',
    'natural yogurt',
    'frozen vegetables',
    'grass-fed beef',
    'free-range chicken',
    'fresh salmon fillet',
    'aromatic coffee beans',
    'pure honey',
    'golden apple',
    'red fruit'
]

# Create a list of unique IDs for each text item in the 'texts' array
 # Each ID follows the format 'food_<index>', where <index> starts from 1
ids = [f"food_{index + 1}" for index, _ in enumerate(texts)]
