from dataset import TrainingSet, ValidationSet
from multiprocessing.pool import ThreadPool


import sys, cv2, os.path as osp, shutil
dataset_path = sys.argv[1]
trn = TrainingSet(dataset_path)
val = ValidationSet(dataset_path)


def convert_label(example, image_save_to, label_save_to):
    image_path, boxes, _ = example
    image = cv2.imread(image_path)
    image_name = osp.basename(image_path)
    new_image_path = osp.join(image_save_to, image_name)
    shutil.copy(image_path, new_image_path)
    height, width, _ = image.shape
    with open(osp.join(label_save_to, image_name.split('.')[0]+'.txt'), 'w') as fw:
        for l,t,w,h in boxes:
            cx = l + w/2
            cy = t + h/2
            print('{} {} {} {} {}'.format(0, cx/width, cy/height, w/width, h/height), file=fw)
    return new_image_path

def save_labels(dataset, image_save_to='images', label_save_to='labels'):
    from tqdm import tqdm

    pool = ThreadPool(16)
    number = len(dataset)
    with tqdm(total=number) as pbar:
        def foo(x):
            pbar.update()
            return convert_label(x, image_save_to, label_save_to)
        image_list = pool.map(foo, dataset)
    pool.close()
    pool.join()
    return image_list

with open('train.txt', 'w') as fw:
    for p in save_labels(trn):
        print(p, file=fw)
with open('valid.txt', 'w') as fw:
    for p in save_labels(val):
        print(p, file=fw)
