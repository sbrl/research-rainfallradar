"use strict";

import fs from 'fs';

import settings from "../../settings.mjs";

import jsonl_to_tf from '../../lib/record/jsonl_to_tf.mjs';

export default async function() {
	if(typeof settings.source !== "string")
		throw new Error(`Error: No source directory specified (see the --source CLI argument)`);
	if(typeof settings.target !== "string")
		throw new Error(`Error: No target directory specified (see the --target CLI argument)`);
	
	if(!fs.existsSync(settings.source))
		throw new Error(`Error: The source directory at '${settings.source}' doesn't exist or you haven't got permission to access it.`);
	if(!fs.existsSync(settings.target))
		await fs.promises.mkdir(settings.target);
	
	// Reordering CLEAN, does NOT shuffle
	await jsonl_to_tf(settings.source, settings.target);
}
