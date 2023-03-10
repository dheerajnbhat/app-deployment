import subprocess
import time

from sanic import Sanic, response

import app as user_src

# We do the model load-to-CPU step on server startup
# so the model object is available globally for reuse
user_src.init()

# Create the http server app
server = Sanic("my_app")


# Healthchecks verify that the environment is correct on Banana Serverless
@server.route("/healthcheck", methods=["GET"])
def healthcheck(request):
    # dependency free way to check if GPU is visible
    gpu = False
    out = subprocess.run("nvidia-smi", shell=True)
    if out.returncode == 0:  # success state on shell command
        gpu = True

    return response.json({"state": "healthy", "gpu": gpu})


# Inference POST handler at '/' is called for every http call from Banana
@server.route("/", methods=["POST"])
def inference(request):
    try:
        model_inputs = response.json.loads(request.json)
    except:
        model_inputs = request.json

    start = time.time()
    output = user_src.inference(model_inputs)
    response_time = time.time() - start

    result = {"class": str(output), "response_time": response_time}

    return response.json(result)


if __name__ == "__main__":
    print("THIS IS FROM THE SERVER")
    server.run(host="0.0.0.0", port=8000, workers=1)
