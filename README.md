# WIDER challenge 2019 pedestrian detection

This project helps with managing the WIDER challenge data of pedestrian detection.

## install

[download](https://competitions.codalab.org/competitions/22852#participate) the data and organize them as the following:
```
.
├── ad_train
│   ├── ad_01
│   ├── ad_02
│   └── ad_03
├── Annotations
├── sur_train
└── val_data
```

## usage  

```python
from dataset import Dataset
from random import choice
import cv2

ds = Dataset()

image_path = choice(ds.image_paths['train'])
image_data = cv2.imread(image_path)
boxes = ds.box['train'][image_path]
ignore_boxes = ds.ignore_box['train'][image_path]
```
