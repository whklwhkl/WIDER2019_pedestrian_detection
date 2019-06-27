from dataset import TrainingSet
from PIL import Image
from tqdm import tqdm_notebook, tqdm
import numpy as np
import cv2


tr = TrainingSet()

objects_per_image = []
for path, boxes, ignore_boxes in tqdm(tr):
    objects_per_image.append(len(boxes))
plt.hist(objects_per_image)
max(objects_per_image)

l,t,w,h,W,H = [[]for i in range(6)]
for path, boxes, ignore_boxes in tqdm(tr):
    if len(boxes):
        try:
            img = Image.open(path)
            img.load()
            img = np.array(img)
            # img = cv2.imread(path)
            height, width, channel = img.shape
            W += [width]
            H += [height]
            left, top, width, height = zip(*boxes)
            l += left
            t += top
            w += width
            h += height
        except Exception as e:
            print(path)
            print(e)
plt.hist(t)
plt.hist(l)
plt.hist(np.log(w), 30, log=True)
np.exp((0.8007598947555256, 0.8467545257117721))
np.exp(np.log(h).mean())
np.log(h).std(), np.log(w).std()
plt.hist(np.log(h), 30, log=1)
plt.hist(W, log=1)
plt.hist(H, log=1)
plt.hist2d(W,H, 32)

uw = np.unique(W)
uh = np.unique(H)
uw
uh
heatmap = np.zeros([len(W), len(H)])
H[:,None]== uw
for i in range(10):
    idx = np.random.choice(np.arange(len(W)))
    print(W[idx], H[idx])

from matplotlib import pyplot as plt
from matplotlib import patches

def show(index):
    img, box, ibox = tr[index]
    img_data = plt.imread(img)

    fig, ax = plt.subplots(1);
    ax.imshow(img_data)

    rects = [patches.Rectangle((x,y), w, h, linewidth=1, edgecolor='r', facecolor='none') for x,y,w,h in ibox]
    for r in rects:
        ax.add_patch(r)
    people = [patches.Rectangle((x,y), w, h, linewidth=1, edgecolor='g', facecolor='none') for x,y,w,h in box]
    for p in people:
        ax.add_patch(p)
    fig.savefig(f'images/demo{index}.png')

for i in range(10):
    show(i)
