import sys
import os.path
import glob
import cv2
import numpy as np
import torch
import architecture as arch
import base64
import json
import requests

model_path = "models/RRDB_ESRGAN_x4.pth"  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
# device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
device = torch.device('cpu')


model = arch.RRDB_Net(3, 3, 64, 23, gc=32, upscale=4, norm_type=None, act_type='leakyrelu', \
                        mode='CNA', res_scale=1, upsample_mode='upconv')
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
for k, v in model.named_parameters():
    v.requires_grad = False
model = model.to(device)

print('Model path {:s}. \nTesting...'.format(model_path))




def image_enhance(model, imgs):
    """
    Input: 
    - imgs : (np.ndarray) of shape (n, d). n is the number of data in this batch
             d is the length of the bytes as numpy int8 array.  
    Output:
    - imgs : (np.ndarray) of shape (n, e)
    """
    import base64
    import io
    import os
    import tempfile
  
    num_imgs = len(imgs)
    upscaled = []
    for i in range(num_imgs):
        # Create a temp file to write to
        tmp = tempfile.NamedTemporaryFile('wb', delete=False, suffix='.png')
        tmp.write(io.BytesIO(imgs[i]).getvalue())
        tmp.close()
        
        # Use PIL to read in the file and compute size

        path = tmp.name

        base = os.path.splitext(os.path.basename(path))[0]
        print(i, base)
        # read image
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = img * 1.0 / 255
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img.unsqueeze(0)
        img_LR = img_LR.to(device)
        print("running on ESRGAN model")
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round()
        
        tmpout = tempfile.NamedTemporaryFile('wb', delete=False, suffix='.png')
        tmpout.close()
        cv2.imwrite(tmpout.name, output)
        
        with open(tmpout.name, "rb") as f:
            data = f.read()
            data_base64 = data.encode("base64")
        
            upscaled.append(data_base64)

         # Remove the temp file
        os.unlink(tmp.name) 
        os.unlink(tmpout.name) 

    return upscaled


from clipper_admin import ClipperConnection, DockerContainerManager
from clipper_admin.deployers.pytorch import deploy_pytorch_model
from torch import nn

clipper_conn = ClipperConnection(DockerContainerManager())
clipper_conn.stop_all()
# clipper_conn.connect()
clipper_conn.start_clipper()


clipper_conn.register_application(name="superresolution", input_type="bytes", default_output="undefined", slo_micros=100000)


print("going to deploy...")

deploy_pytorch_model(
    clipper_conn,
    name="superresolution-model",
    version=1,
    input_type="bytes",
    func=image_enhance,
    pytorch_model=model,
    pkgs_to_install=['opencv-python','numpy']
    )


print("linking model to app...")

clipper_conn.link_model_to_app(app_name="superresolution", model_name="superresolution-model")

def query(addr, filename):
    url = "http://%s/superresolution/predict" % addr
    req_json = json.dumps({
        "input":
        base64.b64encode(open(filename, "rb").read()).decode() # bytes to unicode
    })
    headers = {'Content-type': 'application/json'}
    r = requests.post(url, headers=headers, data=req_json)
    print(r.json())


print("deployed... do a query")

query(clipper_conn.get_query_addr(),'LR/baboon.png')