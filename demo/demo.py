import pdb
pdb.set_trace()

import numpy as np
import torch

print("Torch version:", torch.__version__)

assert torch.__version__.split(".") >= ["1", "7", "1"], "PyTorch 1.7.1 or later is required"
import clip

print(clip.available_models())

model, preprocess = clip.load("ViT-L/14")
if torch.cuda.is_available():
    model.cuda().eval()
else:
    model.eval()
input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size

print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
print("Input resolution:", input_resolution)
print("Context length:", context_length)
print("Vocab size:", vocab_size)

clip.tokenize("Hello World!")

import os
import skimage
import IPython.display
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from collections import OrderedDict
import torch

# images in skimage to use and their textual descriptions
descriptions = {
    "page": "a page of text about segmentation",
    "chelsea": "a facial photo of a tabby cat",
    "astronaut": "a portrait of an astronaut with the American flag",
    "rocket": "a rocket standing on a launchpad",
    "motorcycle_right": "a red motorcycle standing in a garage",
    "camera": "a person looking at a camera on a tripod",
    "horse": "a black-and-white silhouette of a horse", 
    "coffee": "a cup of coffee on a saucer"
}

original_images = []
images = []
titles = []
plt.figure(figsize=(16, 5))

# path = skimage.data_dir
from pathlib import Path
path = Path("./") / "../data/modified_ad_units_jsons/"

for filename in [filename for filename in os.listdir(path) if filename.endswith(".png") or filename.endswith(".jpg")]:
    name = os.path.splitext(filename)[0]
    image = Image.open(os.path.join(path, filename)).convert("RGB")
  
    if len(images) < 8:
        plt.subplot(2, 4, len(images) + 1)
        plt.imshow(image)
        plt.title(f"{filename}\n")
        plt.xticks([])
        plt.yticks([])

    titles.append(name)
    original_images.append(image)
    images.append(preprocess(image))

plt.tight_layout()