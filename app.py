from model import OnnxClassifier


# Init is ran on server startup
# Load your model to CPU as a global variable here using the variable name "model"
def init():
    global model

    device = "cpu"
    model = OnnxClassifier("models/onnx_model.onnx")


# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs: dict) -> dict:
    global model

    # Parse out your arguments
    image = model_inputs.get("image", None)
    if image is None:
        return {"message": "No image provided"}

    # Run the model
    result = model(image)

    # Return the results as a dictionary
    return result
