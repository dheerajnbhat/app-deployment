import base64
import io

import numpy as np
import onnxruntime
from PIL import Image
from torchvision import transforms


class OnnxClassifier:
    def __init__(self, model_path):
        # load onnx model
        self.ort_session = onnxruntime.InferenceSession(model_path)

    def unserialize(self, b64_encoded_img):
        b64_decoded_img = base64.b64decode(b64_encoded_img)
        img = Image.open(io.BytesIO(b64_decoded_img))
        return img

    def preprocess_numpy(self, img):
        resize = transforms.Resize((224, 224))  # must be same as here
        crop = transforms.CenterCrop((224, 224))
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        img = resize(img)
        img = crop(img)
        img = to_tensor(img)
        img = normalize(img)
        return img.unsqueeze(0)

    def predict(self, b64_encoded_img):
        img = self.unserialize(b64_encoded_img)
        input_x = self.preprocess_numpy(img)
        # compute ONNX Runtime output prediction
        ort_inputs = {self.ort_session.get_inputs()[0].name: input_x.cpu().numpy()}
        ort_outs = self.ort_session.run(None, ort_inputs)
        ort_outs = np.argmax(ort_outs)
        return ort_outs


# # load model class
# # from model import OnnxClassifier
# onnx_classifier = OnnxClassifier("models/onnx_model.onnx")

# # get the images
# files = ["data/n01667114_mud_turtle.jpeg", "data/n01440764_tench.jpeg"]
# for file in files:
#     with open(file, "rb") as fp:
#         im_b64 = base64.b64encode(fp.read())
#     # predict output
#     print("Output class for image 1 [turtle]:", onnx_classifier.predict(im_b64))
