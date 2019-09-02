DATASET=$(pwd)
TYPE=person

yolo:
	mkdir labels
	mkdir images
	PYTHONPATH=. python tools/to_image_folder_and_annotation_folder.py ${DATASET}
	echo ${TYPE} > classes.names
