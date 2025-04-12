import chromadb

client = chromadb.PersistentClient()
collection = client.get_or_create_collection(name="videos")

def add_embedding(embedding, filepath: str, timestamp: int):
    collection.add(
        embeddings=tensor_to_embedding(embedding),
        metadatas=[{
            "filepath": filepath,
            "timestamp": timestamp
        }],
        ids=[str(hash(f"{filepath}_{timestamp}"))]
    )

def tensor_to_embedding(tensor):
    return tensor.tolist()
