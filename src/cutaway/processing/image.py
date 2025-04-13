import cv2 as cv
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from database import chroma

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Instantiate model
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)


def embed_video_frames(path: str) -> None:
    cap = cv.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if cap.get(cv.CAP_PROP_FRAME_TYPE) != "I":
            continue

        timestamp = cap.get(cv.CAP_PROP_POS_MSEC)

        # resize to 224 x 224, cropping
        # TODO: experiment with different interpolation methods
        frame = cv.resize(frame, (224, 224), interpolation=cv.INTER_AREA)

        with torch.no_grad():
            embedding = model({ModalityType.VISION: frame})
            chroma.add_embedding(embedding, path, timestamp)
    cap.release()
    cv.destroyAllWindows()
