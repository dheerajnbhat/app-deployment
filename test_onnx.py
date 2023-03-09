import numpy as np
import onnxruntime
import pytest
import torch
from PIL import Image

from pytorch_model import BasicBlock, Classifier


class TestOnnxModel:
    # Fixtures
    @pytest.fixture
    def torch_model(self):
        torch_model = Classifier(BasicBlock, [2, 2, 2, 2])
        torch_model.load_state_dict(torch.load("models/pytorch_model_weights.pth"))
        torch_model.eval()
        return torch_model

    @pytest.fixture
    def ort_session(self):
        ort_session = onnxruntime.InferenceSession("models/onnx_model.onnx")
        return ort_session

    # function to test if outputs of image are belonging to same class or not
    # from both pytorch and onnx model
    @pytest.mark.parametrize(
        "test_input",
        [
            Image.open("data/n01667114_mud_turtle.jpeg"),
            Image.open("data/n01440764_tench.jpeg"),
        ],
    )
    def test_model_output(self, torch_model, ort_session, test_input):
        input_x = torch_model.preprocess_numpy(test_input).unsqueeze(0)
        # pytorch model output
        torch_out = torch_model.forward(input_x)
        torch_out = torch.argmax(torch_out)
        # onnx model output
        ort_inputs = {ort_session.get_inputs()[0].name: input_x.cpu().numpy()}
        ort_outs = ort_session.run(None, ort_inputs)
        ort_outs = np.argmax(ort_outs)
        assert ort_outs == torch_out
