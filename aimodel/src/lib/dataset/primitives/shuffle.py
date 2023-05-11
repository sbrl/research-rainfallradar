import os

from copy import deepcopy
import random

from loguru import logger


def shuffle(lst):
	"""
	Shuffles a list with the Fisher-Yates algorithm.
	@ref	https://poopcode.com/shuffle-a-list-in-python-fisher-yates/
	@param	lst		list	The list to shuffle.
	@return	list	The a new list that is a shuffled copy of the original.
	"""

	tmplist = deepcopy(lst)
	m = len(tmplist)
	
	if "RANDSEED" in os.environ:
		seed = os.environ["RANDSEED"]
		random.seed(seed)
		logger.info(f"Random seed set to {seed}, first 3 values: {random.randint(0, m)}, {random.randint(0, m)}, {random.randint(0, m)}")
	
	while (m):
		m -= 1
		i = random.randint(0, m)
		tmplist[m], tmplist[i] = tmplist[i], tmplist[m]
	return tmplist
