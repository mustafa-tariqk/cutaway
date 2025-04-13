from imagebind import ModalityType
from imagebind.models.imagebind_model import (
    ImageBindModel,
    SimpleNamespace,
    imagebind_huge,
)
from typing import Dict
from torch import Tensor, cuda, no_grad


class Model:
    def __init__(self):
        self.model: ImageBindModel = imagebind_huge(pretrained=True).eval()
        self.model.to("cuda:0" if cuda.is_available() else "cpu")

    def embed_video_frames(self, video_frames: Tensor):
        data: Dict[SimpleNamespace, Tensor] = {ModalityType.VISUAL: video_frames}
        with no_grad():
            return self.model(data)

    def embed_video_audio(self, audio_chunks: Tensor):
        data: Dict[SimpleNamespace, Tensor] = {ModalityType.AUDIO: audio_chunks}
        with no_grad():
            return self.model(data)
