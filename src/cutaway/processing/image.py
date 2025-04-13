import cv2 as cv
import torch
from torch import Tensor
from torchvision.transforms import (
    Compose,
    Resize,
    InterpolationMode,
    CenterCrop,
    ToTensor,
    Normalize,
)
from PIL import Image
from model.imagebind import Model


class ImageProcessor:
    def __init__(self):
        self.model: Model = Model()
        self.transform: Compose = Compose(
            [
                Resize(224, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(224),
                ToTensor(),
                Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    def embed_video(self, path: str) -> Tensor:
        frames = []
        cap = cv.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if cap.get(cv.CAP_PROP_FRAME_TYPE) != "I":
                continue

            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            processed_frame = self.transform(pil_image)
            frames.append(processed_frame)

        cap.release()
        cv.destroyAllWindows()

        if not frames:
            return torch.zeros((0, 512))

        # Stack all frames into a single tensor
        frames_tensor = torch.stack(frames, dim=0)
        return self.model.embed_video_frames(frames_tensor)
