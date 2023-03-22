import os
import seaborn as sns
import pandas as pd

def plot_metric(train, val, name, dir_output):
	plt.plot(train, label=f"train_{name}")
	plt.plot(val, label=f"val_{name}")
	plt.title(name)
	plt.xlabel("epoch")
	plt.ylabel(name)
	plt.savefig(os.path.join(dir_output, f"{name}.png"))
	plt.close()

FILEPATH_INPUT = os.environ["INPUT"]
DIRPATH_OUTPUT = os.environ["OUTPUT"] if "OUTPUT" in os.environ else os.getcwd()


df = pd.read_csv(FILEPATH_INPUT)

