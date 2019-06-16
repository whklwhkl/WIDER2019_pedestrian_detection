from dataset import TrainingSet


tr = TrainingSet()

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
    fig.savefig(f'demo{index}.png')

for i in range(10):
    show(i)
