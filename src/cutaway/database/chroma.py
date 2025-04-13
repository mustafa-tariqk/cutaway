from typing import List

from chromadb.api.models.Collection import Collection
from chromadb.base_types import Vector
from chromadb import PersistentClient


class DB:
    def __init__(self, path_to_videos: str):
        self.collection: Collection = PersistentClient().get_or_create_collection(
            name=path_to_videos
        )

    def add_video_embedding(
        self, embeddings: Vector, filepath: str, timestamps: List[float]
    ):
        self.collection.add(
            embeddings=embeddings,
            metadatas=[
                {"filepath": filepath, "timestamp": timestamp}
                for timestamp in timestamps
            ],
            ids=[str(hash(f"{filepath}_{timestamp}")) for timestamp in timestamps],
        )
