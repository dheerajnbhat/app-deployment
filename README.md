# App deployment for image classification

This repository contains the codebase to successfully deploy image classification model to banana platform

This repository has following structure:
  - data (folder which contains sample images for testing purposes)
  - models (folder which contains pytorch model, trained on (ImageNet Dataset)[https://www.image-net.org] and onnx model converted from pytorch model)
  - pytorch_model.py (pytorch model code)
  - convert_to_onnx.py (code to convert pytorch model to onnx model)
  - test_onnx.py (test cases using pytest to verify iutputs from pytorch and onnx model)
  - model.py (model loading and inference code)
  - test_server.py (code to make a call to the model deployed on the banana dev)
  - requirements.txt (python dependencies to be installed)
  - Dockerfile (to build docker container)

## Getting Started

You can clone the repository using the command:
```
git clone https://github.com/dheerajnbhat/app-deployment.git
cd app-deployment
```

Now install the python dependencies (code successfully run with python 3.8) using the following command:
```
pip3 install -r requirements.txt
```

Now you can run the following files using the command:
```
python3 convert_to_onnx.py
py.test test_onnx.py
```

To run the inference on sample images, run the following code:
```
import base64
from model import OnnxClassifier, preprocess_numpy

# load model class
onnx_classifier = OnnxClassifier("models/onnx_model.onnx")

# get the images
files = ["data/n01667114_mud_turtle.jpeg", "data/n01440764_tench.jpeg"]
for file in files:
    with open(file, "rb") as fp:
        im_b64 = base64.b64encode(fp.read())
    # predict output
    print("Output class for image 1 [turtle]:", onnx_classifier.predict(im_b64))
```
