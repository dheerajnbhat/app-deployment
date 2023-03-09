import numpy as np
import onnxruntime
from PIL import Image
from torchvision import transforms


def preprocess_numpy(img):
    resize = transforms.Resize((224, 224))  # must be same as here
    crop = transforms.CenterCrop((224, 224))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    img = resize(img)
    img = crop(img)
    img = to_tensor(img)
    img = normalize(img)
    return img.unsqueeze(0)


class OnnxClassifier:
    def __init__(self, model_path):
        # load onnx model
        self.ort_session = onnxruntime.InferenceSession(model_path)

    def predict(self, input_x):
        # compute ONNX Runtime output prediction
        ort_inputs = {self.ort_session.get_inputs()[0].name: input_x.cpu().numpy()}
        ort_outs = self.ort_session.run(None, ort_inputs)
        ort_outs = np.argmax(ort_outs)
        return ort_outs


# # load class
# onnx_classifier = OnnxClassifier("models/onnx_model.onnx")

# # get the images
# img1 = Image.open("data/n01667114_mud_turtle.jpeg")
# img1 = preprocess_numpy(img1)
# img2 = Image.open("data/n01440764_tench.jpeg")
# img2 = preprocess_numpy(img2)

# # predict output
# print("Output class for image 1 [turtle]:", onnx_classifier.predict(img1))
# print("Output class for image 2 [tench]:", onnx_classifier.predict(img2))
