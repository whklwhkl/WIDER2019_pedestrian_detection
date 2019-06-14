import os.path as osp


SPLIT = ['train', 'val', 'test']
TYPE = ['list', 'ignore', 'bbox']
CATEGORIES = ['pedestrian', 'ignore']
CAT_INFO = [{'name': c, 'id': i + 1} for i, c in enumerate(CATEGORIES)]


class _Dataset:
	def __init__(self, data_dir=''):
		self.anno_pattern = osp.join(data_dir, 'Annotations', '{split}_{type}.txt')
		self.image_paths = None
		self.ignore_box = None
		self.box = None
		self.length = 0

	def __getitem__(self, index):
		return self.image_paths[index], self.box[index], self.ignore_box[index]

	def __len__(self):
		return self.length

	def load(self):
		self.image_paths = list(self._read_txt(TYPE[0]).keys())
		self.ignore_box = self._read_txt(TYPE[1])
		self.box = self._read_txt(TYPE[2])
		boxes, ignore_boxes = [], []
		for ip in self.image_paths:
			boxes += [[] or self.box.get(ip)]
			ignore_boxes += [[] or self.ignore_box.get(ip)]
		self.box = boxes
		self.ignore_box = ignore_boxes
		self.length = len(self.image_paths)

	def _read_txt(self, type):
		image_paths = []
		bboxes = []
		txt_path = self.anno_pattern.format(**{'type':type})
		with open(txt_path) as f:
			for line in f:
				tokens = line.strip().split()
				name = tokens[0]
				image_paths += [self._extend_path(name)]
				bbox = []
				for i in range(len(tokens)//4):
					start = i*4 + 1
					bbox += [list(float(x) for x in tokens[start:start+4])]
				bboxes += [bbox]
		return dict(zip(image_paths, bboxes))

	def _extend_path(self, image_name):
		raise NotImplementedError('define name extending logic')


class TrainingSet(_Dataset):
	def __init__(self, data_dir=''):
		super().__init__()
		self.anno_pattern = self.anno_pattern.format(split=SPLIT[0], type='{type}')
		self.sur = osp.join(data_dir, 'sur_train', '{}')
		self.ad = osp.join(data_dir, 'ad_train', 'ad_0{}', '{}')
		self.load()

	def _extend_path(self, image_name):
		if image_name.startswith('sur'):
			return self.sur.format(image_name)
		else:
			for i in range(1,4):
				guess_path = self.ad.format(i, image_name)
				if osp.exists(guess_path):
					return guess_path
		raise KeyError('cannot locate image')


class ValidationSet(_Dataset):
	def __init__(self, data_dir=''):
		super().__init__()
		self.anno_pattern = self.anno_pattern.format(split=SPLIT[1], type='{type}')
		self.val = osp.join(data_dir, 'val_data', '{}')
		self.load()

	def _extend_path(self, image_name):
		return self.val.format(image_name)


class TestSet(_Dataset):
	def __init__(self, data_dir=''):
		super().__init__()

	def load(self):
		self.image_paths = list(self._read_txt(TYPE[0]).keys())


if __name__ == '__main__':
	training = TrainingSet()
	validation = ValidationSet()
