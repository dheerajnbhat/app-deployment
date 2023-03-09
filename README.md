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
from model import OnnxClassifier, preprocess_numpy

# load model class
onnx_classifier = OnnxClassifier("models/onnx_model.onnx")

# get the images
img1 = Image.open("data/n01667114_mud_turtle.jpeg")
img1 = preprocess_numpy(img1)
img2 = Image.open("data/n01440764_tench.jpeg")
img2 = preprocess_numpy(img2)

# predict output
print("Output class for image 1 [turtle]:", onnx_classifier.predict(img1))
print("Output class for image 2 [tench]:", onnx_classifier.predict(img2))
```
