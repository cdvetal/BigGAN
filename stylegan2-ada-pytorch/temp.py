import pickle

import pygame
import torch
from torchvision import transforms

from superresolution.SRGAN import srresnet_x2, load_pretrained_state_dict

pygame.init()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "superresolution/SRGAN_x2-SRGAN_ImageNet.pth.tar"
sr_model = srresnet_x2(in_channels=3,out_channels=3, channels=64, num_rcb=16)
sr_model = load_pretrained_state_dict(sr_model, False, model_path)

# Start the verification mode of the model.
sr_model.eval().cuda()
# Turn on half-precision inference.
# model.half()

# Set up the drawing window
screen = pygame.display.set_mode([1024, 1024])

with open('network-snapshot-000260.pkl', 'rb') as f:
    G = pickle.load(f)['G_ema'].eval().cuda()  # torch.nn.Module


# z = torch.lerp(z_init, z_end, i / n_images)

trans = transforms.Compose([
    transforms.RandomInvert(p=1.0),
    # transforms.Resize((1024, 1024), antialias=True),
    transforms.ToPILImage(),
])

clock = pygame.time.Clock()

c = None                                # class labels (not used in this example)
i = 1
n_images = 10
current_z = last_z = torch.randn([1, G.z_dim]).cuda()
target_z = torch.randn([1, G.z_dim]).cuda()

with torch.no_grad():
    # Run until the user asks to quit
    running = True
    while running:
        # Did the user click the window close button?
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Fill the background with white
        screen.fill((255, 255, 255))

        img = G(current_z, c).squeeze(0)

        img = torch.clamp(img, min=0.0, max=1.0)

        # img = sr_model(img).squeeze(0)

        img = trans(img)

        raw_str = img.tobytes("raw", 'RGBA')
        pygame_img = pygame.image.fromstring(raw_str, (1024, 1024), 'RGBA')

        screen.blit(pygame_img, (0, 0))

        # Flip the display
        pygame.display.flip()

        # print(i)
        i += 1

        if i > 0 and i % n_images == 0:
            last_z = target_z
            target_z = torch.randn([1, G.z_dim]).cuda()
            print("Changed")
        else:
            current_z = torch.lerp(last_z, target_z, (i % n_images) / n_images)
            print((i % n_images) / n_images)

        clock.tick(5)

# Done! Time to quit.
pygame.quit()




