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
from dataset import TrainingSet, ValidationSet, TestSet  # all have similar defs

tr = TrainingSet()    # take TrainingSet for example
len(tr) == 91500      # __len__ method implemented

image_path, boxes, ignore_boxes = tr[0]   # __getitem__
# ('sur_train/sur00005.jpg',      # for TestSet, only image path is returned

# [[1209.0, 538.0, 60.0, 167.0],  # each box conforms [left, top, width, height]
#  [1094.0, 419.0, 29.0, 104.0],
#  [1187.0, 415.0, 32.0, 99.0],
#  [1226.0, 415.0, 26.0, 100.0],
#  [1253.0, 423.0, 37.0, 89.0],
#  [1239.0, 347.0, 20.0, 53.0],
#  [1193.0, 356.0, 17.0, 49.0],
#  [1167.0, 358.0, 30.0, 43.0],
#  [1232.0, 343.0, 12.0, 47.0],
#  [1215.0, 337.0, 15.0, 44.0],
#  [1151.0, 350.0, 13.0, 38.0],
#  [1137.0, 344.0, 14.0, 39.0],
#  [1294.0, 338.0, 20.0, 43.0]],

# [[886.0, 400.0, 98.0, 208.0],  # if there's no box, [] is returned
# [999.0, 274.0, 259.0, 206.0],
# [1258.0, 259.0, 213.0, 249.0],
# [1466.0, 250.0, 280.0, 171.0]])
```
