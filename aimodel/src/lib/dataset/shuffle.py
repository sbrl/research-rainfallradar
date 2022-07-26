from copy import deepcopy
from random import randint


def shuffle(lst):
	"""
	Shuffles a list with the Fisher-Yates algorithm.
	@ref	https://poopcode.com/shuffle-a-list-in-python-fisher-yates/
	@param	lst		list	The list to shuffle.
	@return	list	The a new list that is a shuffled copy of the original.
	"""
	tmplist = deepcopy(lst)
	m = len(tmplist)
	while (m):
		m -= 1
		i = randint(0, m)
		tmplist[m], tmplist[i] = tmplist[i], tmplist[m]
	return tmplist
