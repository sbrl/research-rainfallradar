import io

from .handle_open import handle_open

def readfile(filepath, transparent_gzip=True):
	handle = io.open(filepath, "r") if transparent_gzip else handle_open(filepath, "r")
	content = handle.read()
	handle.close()
	return content