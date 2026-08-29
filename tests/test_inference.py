import pytest
import torch
from src.model import CatDogCNN


class TestModelInference:

    def setup_method(self):
        self.tModel = CatDogCNN()
        self.tModel.eval()

    def test_model_output_shape(self):
        tDummy_Input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            tOutput = self.tModel(tDummy_Input)
        assert tOutput.shape == (1, 1)

    def test_output_between_0_and_1(self):
        tDummy_Input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            tOutput = self.tModel(tDummy_Input)
        assert 0 <= tOutput.item() <= 1

    def test_batch_inference(self):
        tBatch_Input = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            tOutput = self.tModel(tBatch_Input)
        assert tOutput.shape == (4, 1)

    def test_model_deterministic_eval(self):
        tDummy_Input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            tOut1 = self.tModel(tDummy_Input)
            tOut2 = self.tModel(tDummy_Input)
        assert torch.equal(tOut1, tOut2)