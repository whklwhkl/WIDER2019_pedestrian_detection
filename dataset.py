import os.path as osp
import json


SPLIT = ['train', 'val']
TYPE = ['list', 'ignore', 'bbox']
CATEGORIES = ['pedestrian', 'ignore']


class Dataset:
	def __init__(self, data_dir='', anno_dir='Annotations'):
		self.anno_pattern = osp.join(data_dir, anno_dir, '{split}_{type}.txt')
		self.sur = osp.join(data_dir, 'sur_train', '{}')
		self.ad = osp.join(data_dir, 'ad_train', 'ad_0{}', '{}')
		self.val = osp.join(data_dir, 'val_data', '{}')
		self.cat_info = [{'name':c, 'id':i+1}for i,c in enumerate(CATEGORIES)]
		self.image_paths = {}
		self.ignore_box = {}
		self.box = {}

		for split in SPLIT:
			self.image_paths[split] = list(self._read_txt(split, TYPE[0]).keys())
			self.ignore_box[split] = self._read_txt(split, TYPE[1])
			self.box[split] = self._read_txt(split, TYPE[2])

	def to_json(self, dest_path):
		meta = {'images':[], 'annotations':[], 'cat_info':self.cat_info}
		for split in SPLIT:
			for path in self.image_paths[split]:
				image_info = {'file_name':path}
				annot_info = {}
		json.dump(meta, open(dest_path,'wb'))

	def _read_txt(self, split, type):
		image_paths = []
		bboxes = []
		txt_path = self.anno_pattern.format(**{'split':split, 'type':type})
		with open(txt_path) as f:
			for line in f:
				tokens = line.strip().split()
				name = tokens[0]
				image_paths += [self._extend_path(name, split)]
				bbox = []
				for i in range(len(tokens)//4):
					start = i*4 + 1
					bbox += [list(float(x) for x in tokens[start:start+4])]
				bboxes += [bbox]
		return dict(zip(image_paths, bboxes))

	def _extend_path(self, image_name, split):
		if split == SPLIT[0]:
			if image_name.startswith('sur'):
				return self.sur.format(image_name)
			else:
				for i in range(1,4):
					guess_path = self.ad.format(i, image_name)
					if osp.exists(guess_path):
						return guess_path
		elif split == SPLIT[1]:
			return self.val.format(image_name)
		raise KeyError('cannot locate image')


if __name__ == '__main__':
	wpd = Dataset()
	# print(l12[:5], b12[:5])
