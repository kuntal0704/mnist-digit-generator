
import streamlit as st
import torch, os
from PIL import Image
import numpy as np

st.set_page_config(page_title="MNIST Digit Generator")

@st.cache_resource
def load_artifacts():
    device='cpu'
    latent_dim=16
    # define decoder identical to training
    class Decoder(torch.nn.Module):
        def __init__(self, latent_dim=16):
            super().__init__()
            self.fc = torch.nn.Sequential(
                torch.nn.Linear(latent_dim,128), torch.nn.ReLU(),
                torch.nn.Linear(128,7*7*64), torch.nn.ReLU())
            self.deconv = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(64,32,3,2,1,output_padding=1), torch.nn.ReLU(),
                torch.nn.ConvTranspose2d(32,1,3,2,1,output_padding=1), torch.nn.Sigmoid())
        def forward(self,z):
            h = self.fc(z).view(-1,64,7,7)
            return self.deconv(h)
    decoder = Decoder(latent_dim).to(device)
    state_dict = torch.load(os.path.join('artifacts','ae.pth'),map_location=device)
    # keep only decoder parameters
    decoder.load_state_dict({k.replace('decoder.',''):v for k,v in state_dict.items() if k.startswith('decoder.')})
    decoder.eval()
    mean_latents = torch.load(os.path.join('artifacts','mean_latents.pth'),map_location=device)
    return decoder, mean_latents, device

decoder, mean_latents, device = load_artifacts()

st.title("Hand‑written Digit Generator")

digit = st.selectbox("Select digit to generate", list(range(10)), format_func=str)
generate = st.button("Generate 5 images")

def sample_digit(digit, n=5, noise_scale=0.8):
    mu = mean_latents[digit]
    z = mu + noise_scale*torch.randn(n, mu.size(0))
    with torch.no_grad():
        imgs = decoder(z.to(device)).cpu()
    return imgs

if generate:
    imgs = sample_digit(digit,5)
    cols = st.columns(5)
    for col, img in zip(cols, imgs):
        arr = (img.squeeze().numpy()*255).astype(np.uint8)
        col.image(Image.fromarray(arr), width=100,caption=f'{digit}')

st.markdown("""
---
### How it works
* The model is a lightweight convolutional autoencoder trained **from scratch** on MNIST.
* At inference time we sample around each digit's mean latent vector to produce diverse yet recognizable images.
* Training script and model weights are available in the repository.
""")
