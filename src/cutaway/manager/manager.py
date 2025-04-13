from pathlib import Path
import cv2
from database.chroma import DB
from processing.image import ImageProcessor


class Manager:
    def __init__(self, path_to_videos: str):
        self.path = path_to_videos
        self.db: DB = DB(path_to_videos)
        self.image_proc: ImageProcessor = ImageProcessor()
        self.video_files = self.find_video_files()

    def index(self):
        for video_file in self.video_files:
            # embed
            embedding = self.image_proc.embed_video(video_file)
            # add to db
            # TODO: Fix this up
            self.db.add_video_embedding(embedding, video_file, [10])

    def find_video_files(self) -> list[str]:
        root_path = Path(self.path)
        video_files = []

        for file_path in root_path.rglob("*"):
            if file_path.is_file():
                try:
                    cap = cv2.VideoCapture(str(file_path))
                    if cap.isOpened():
                        video_files.append(file_path)
                    cap.release()
                except Exception:
                    continue

        return video_files


if __name__ == "__main__":
    manager = Manager("path/to/videos")
