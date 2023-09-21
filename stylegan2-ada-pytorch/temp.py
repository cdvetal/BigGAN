import os

import torch
from torch import nn
from torchvision import transforms
from torchvision.utils import save_image
import pickle

from superresolution.model import srresnet_x8, srresnet_x4, srresnet_x2
from superresolution.utils import load_pretrained_state_dict
from basicsr.archs.rrdbnet_arch import RRDBNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_weights_path = "superresolution/SRResNet_x4-SRGAN_ImageNet.pth.tar"

"""
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
loadnet = torch.load("esrgan/RealESRGAN_x4plus.pth")
model.load_state_dict(loadnet['params_ema'], strict=True)
model = model.cuda()
netscale = 4
"""

trans = transforms.RandomInvert(p=1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def build_model() -> nn.Module:
    # Initialize the super-resolution model
    sr_model = srresnet_x4()
    print(f"Build `SRResNet` model successfully.")

    # Load model weights
    sr_model = load_pretrained_state_dict(sr_model, False, model_weights_path)
    print(f"Load `SRResNet` model weights `{os.path.abspath(model_weights_path)}` successfully.")

    # Start the verification mode of the model.
    sr_model.eval()

    # Enable half-precision inference to reduce memory usage and inference time
    # if args.half:
    # sr_model.half()

    sr_model = sr_model.to(device)

    return sr_model


# Initialize the model
sr_model = build_model()


n_images = 4

with open('network-snapshot-000161.pkl', 'rb') as f:
    G = pickle.load(f)['G_ema'].eval().cuda()  # torch.nn.Module

"""
z_init = torch.randn([G.z_dim])
z_end = torch.randn([G.z_dim])

zs = []
for i in range(n_images):
    z = torch.lerp(z_init, z_end, i / n_images)
    zs.append(z)

zs = torch.stack(zs, dim=0).cuda()
"""
zs = torch.randn([n_images, G.z_dim]).cuda()
c = None                                # class labels (not used in this example)
with torch.no_grad():
    img = G(zs, c)                           # NCHW, float32, dynamic range [-1, +1]

    print(img.shape)

    save_image(img, f"images/images.png")

    del G

    img = sr_model(img)
    # img = model(img)

    print(img.shape)

    save_image(img, f"images/scaled_images.png")

    img = trans(img)

    save_image(img, f"images/blue_images.png")



