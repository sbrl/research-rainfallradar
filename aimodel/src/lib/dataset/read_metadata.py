import os
import json


from ..io.readfile import readfile

def read_metadata(dirpath_dataset):
	if os.path.isfile(dirpath_dataset):
		filepath_metadata = os.path.join(os.path.dirname(dirpath_dataset), "metadata.jsonl")
	else:
		filepath_metadata = os.path.join(dirpath_dataset, "metadata.json")
	
	return json.loads(readfile(filepath_metadata))