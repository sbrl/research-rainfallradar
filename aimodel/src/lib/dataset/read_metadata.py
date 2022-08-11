import os
import json


from ..io.readfile import readfile

def read_metadata(dirpath_dataset):
	filepath_metadata = os.path.join(dirpath_dataset, "metadata.json")
	
	return json.loads(readfile(filepath_metadata))