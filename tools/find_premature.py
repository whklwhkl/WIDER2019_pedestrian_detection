import sys, os
from PIL import Image

image_folder = sys.argv[1]

from glob import glob

images = glob(os.path.join(image_folder, '*.jpg'))

for p in images:
    try:
        Image.open(p).load()
    except :
        print(p)
