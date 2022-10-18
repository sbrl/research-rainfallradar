"use strict";

import fs from 'fs';

import settings from "../../settings.mjs";

import records_recompress from '../../lib/record/records_recompress.mjs';

export default async function() {
	if(typeof settings.source !== "string")
		throw new Error(`Error: No source directory specified (see the --source CLI argument)`);
	if(typeof settings.target !== "string")
		throw new Error(`Error: No target directory specified (see the --target CLI argument)`);
	
	if(!fs.existsSync(settings.source))
		throw new Error(`Error: The source directory at '${settings.source}' doesn't exist or you haven't got permission to access it.`);
	if(!fs.existsSync(settings.target))
		await fs.promises.mkdir(settings.target);
	
	// Recompressing CLEAN, does NOT shuffle
	await records_recompress(settings.source, settings.target, settings.count_file);
}
