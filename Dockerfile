FROM clipper/python36-closure-container:0.3

RUN pip install https://download.pytorch.org/whl/cpu/torch-1.0.0-cp36-cp36m-linux_x86_64.whl torchvision opencv-python numpy six Pillow wheel certifi

# Copy the module from your local filesystem into the Docker image
COPY ./clipper_service.py ./architecture.py ./block.py ./

