import os
import pickle
import time

import numpy as np
import pygame
import pyaudio
import scipy
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from superresolution.SRGAN import srresnet_x2, load_pretrained_state_dict

# Set up the drawing window position
x, y = 0, 0
os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (x, y)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

pygame.init()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# model_path = "superresolution/SRGAN_x2-SRGAN_ImageNet.pth.tar"
# sr_model = srresnet_x2(in_channels=3, out_channels=3, channels=64, num_rcb=16)
# sr_model = load_pretrained_state_dict(sr_model, False, model_path)

# Start the verification mode of the model.
# sr_model.eval().to(device)


c = None  # class labels (not used in this example)
step = 0
max_steps = 1000
n_images = 1
img_size = 1024

# Set up the drawing window
screen = pygame.display.set_mode([1024 * 1, 1024 * 1]) #, pygame.NOFRAME)

with open('network-snapshot-000260.pkl', 'rb') as f:
    G = pickle.load(f)['G_ema'].eval().to(device)  # torch.nn.Module

# Audio related variables
CHUNK = G.z_dim
RATE = 44100
FORMAT = pyaudio.paInt16
CHANNELS = 1

trans = transforms.Compose([
    transforms.RandomInvert(p=1.0),
    # transforms.Resize((2048, 2048), antialias=False, interpolation=InterpolationMode.NEAREST),
    transforms.ToPILImage(),
])

clock = pygame.time.Clock()

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)

num_keyframes = 100  # Number of seeds to interpolate through.
w_frames = 60  # Number of frames to interpolate between latents

zs = torch.randn(num_keyframes * n_images, G.z_dim).to(device)
ws = G.mapping(z=zs, c=None, truncation_psi=1.0)
ws = ws.reshape(n_images, num_keyframes, *ws.shape[1:])

row = []
for xi in range(n_images):
    x = np.arange(-num_keyframes * 2, num_keyframes * (2 + 1))
    y = np.tile(ws[xi].cpu().numpy(), [2 * 2 + 1, 1, 1])
    interp = scipy.interpolate.interp1d(x, y, kind="cubic", axis=0)
    row.append(interp)

with torch.no_grad():
    # Run until the user asks to quit
    running = True
    while running:
        # Did the user click the window close button?
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # stream.stop_stream()
                    # stream.close()
                    # p.terminate()
                    running = False

        # Fill the background with white
        screen.fill((255, 255, 255))

        audio_data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
        # print(audio_data.shape)
        peak = np.average(np.abs(audio_data)) * 2
        audio_value = int(100*peak/2**16)
        print(audio_value)
        # print("Peak", peak)
        noise = torch.randn(1, G.z_dim).repeat(18, 1).to(device)
        noise *= (audio_value / 10)

        # images = sr_model(images)

        for xi in range(n_images):
            interp = row[xi]
            w = torch.from_numpy(interp(step / w_frames)).to(device)
            w += noise
            w = torch.clamp(w, min=-1.0, max=1.0)

            images = G.synthesis(ws=w.unsqueeze(0), noise_mode='const')[0]

            images = torch.clamp(images, min=0.0, max=1.0)

            # images = sr_model(images.unsqueeze(0)).squeeze()

            images = trans(images)

            raw_str = images.tobytes("raw", 'RGB')
            pygame_img = pygame.image.fromstring(raw_str, (1024, 1024), 'RGB')

            screen.blit(pygame_img, ((1024 * xi), 0))

        # Flip the display
        pygame.display.flip()

        # print(i)
        step += 1

        clock.tick(60)

# Done! Time to quit.
pygame.quit()

