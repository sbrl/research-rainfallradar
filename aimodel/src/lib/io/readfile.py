import io

def readfile(filepath):
	handle = io.open(filepath, "r")
	content = handle.read()
	handle.close()
	return content