import cv2
from glob import glob


SPLIT = ['train', 'val']
TYPE = ['list', 'ignore', 'bbox']
PATTERN = 'Annotations/{split}_{type}.txt'
IMAGE_DIR = '.'


def read_txt(path):
	with open(path) as f:
		pass
	return


if __name__ == '__main__':
	print(PATTERN.format(**{'split':SPLIT[0], 'type':TYPE[0]}))
