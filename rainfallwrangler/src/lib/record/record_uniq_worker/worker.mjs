"use strict";

import workerpool from 'workerpool';

import hash_targets from './hash_targets.mjs';
import delete_duplicates from './delete_duplicates.mjs';

workerpool.worker({
	hash_targets,
	delete_duplicates
});
