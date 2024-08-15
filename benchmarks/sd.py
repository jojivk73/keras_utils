

from PIL import Image

from keras_cv.models import StableDiffusion
from utils import *

args = get_args()
set_precision_and_threads(args)

model = StableDiffusion(img_height=256, img_width=256, jit_compile=True)
img = model.text_to_image(
    "Photograph of a beautiful horse running through a field"
)
Image.fromarray(img[0]).save("horse.png")
print("Saved at horse.png")
