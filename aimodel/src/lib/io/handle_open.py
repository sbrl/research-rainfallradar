import io
import gzip


def handle_open(filepath, mode):
	if filepath.endswith(".gz"):
		return gzip.open(filepath, mode)
	else:
		return io.open(filepath, mode)