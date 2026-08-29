import pytest
from PIL import Image
import torch
import numpy as np
from src.utils import preprocess_image


class TestPreprocessing:

    def test_image_resize_to_224(self):
        tDummy_Img = Image.fromarray(np.random.randint(0, 255, (500, 300, 3), dtype=np.uint8))
        tProcessed = preprocess_image(tDummy_Img)
        assert tProcessed.shape[-2:] == (224, 224)

    def test_output_is_tensor(self):
        tDummy_Img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        tProcessed = preprocess_image(tDummy_Img)
        assert isinstance(tProcessed, torch.Tensor)

    def test_output_has_batch_dimension(self):
        tDummy_Img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        tProcessed = preprocess_image(tDummy_Img)
        assert tProcessed.dim() == 4

    def test_rgb_channels(self):
        tDummy_Img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        tProcessed = preprocess_image(tDummy_Img)
        assert tProcessed.shape[1] == 3