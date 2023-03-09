import onnx
import torch
from PIL import Image

from pytorch_model import BasicBlock, Classifier


def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


if __name__ == "__main__":
    # load torch model
    torch_model = Classifier(BasicBlock, [2, 2, 2, 2])
    torch_model.load_state_dict(torch.load("models/pytorch_model_weights.pth"))
    torch_model.eval()

    # torch export runs the model, so we need an input to convert torch model
    # to onnx model
    img = Image.open("data/n01667114_mud_turtle.jpeg")
    # input
    input_x = torch_model.preprocess_numpy(img).unsqueeze(0)

    # Export the model
    torch.onnx.export(
        torch_model,  # model being run
        input_x,  # model input (or a tuple for multiple inputs)
        "models/onnx_model.onnx",  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=10,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "input": {0: "batch_size"},  # variable length axes
            "output": {0: "batch_size"},
        },
    )

    # check onnx model for valid schema
    onnx_model = onnx.load("models/onnx_model.onnx")
    onnx.checker.check_model(onnx_model)
    print("Torch model successfully exported to Onnx!")
