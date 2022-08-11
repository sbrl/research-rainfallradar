import os

def find_paramsjson(filepath_checkpoint):
	filepath_stem = os.path.splitext(filepath_checkpoint)[0]
	dirpath_container = os.path.dirname(filepath_checkpoint)
	dirpath_parent = os.path.dirname(dirpath_container)
	
	options = [
		f"{filepath_stem}.json",
		os.path.join(dirpath_container, "params.json"),
		os.path.join(dirpath_parent, "params.json")
	]
	for candidate in options:
		if os.path.exists(candidate):
			return candidate
	
	return None