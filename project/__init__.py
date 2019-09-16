import numpy as np
import PIL
from project import models
import torch
from torchvision import transforms
from torch.autograd import Variable


# ====================================== #
#          Optional for PyTorch          #
# ====================================== #
# from .models import Encoder, Decoder
encoder = models.Encoder()
decoder = models.Decoder()
encoder.eval()
decoder.eval()

# Load model weights here
encoder.load_state_dict(torch.load('project/final-ckpt-encoder-199ep'))
decoder.load_state_dict(torch.load('project/final-ckpt-decoder-199ep'))


def encode(img):
    """
    Your code here
    img: a 256x256 PIL Image
    """
    img_transform = transforms.Compose([transforms.ToTensor()])
    x = img_transform(img)
    x = x.unsqueeze(0)

    return np.array(encoder(x).detach().numpy())
    # raise NotImplementedError('encode')

def decode(x):
    """
    Your code here
    x: a numpy array, <= 8192 Byte
    """
    x = torch.from_numpy(x)
    x = decoder(x)
    x = np.moveaxis(x.detach().numpy().squeeze(0), 0, -1)
    return x*255
    # raise NotImplementedError('decode')
