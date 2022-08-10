import io

def writefile(filepath, content):
	handle = io.open(filepath, "w")
	handle.write(content)
	handle.close()