"use strict";

import fs from 'fs';

import settings from "../../settings.mjs";

import RecordUniqManager from '../../lib/record/RecordUniqManager.mjs';

export default async function() {
	if(typeof settings.source !== "string")
		throw new Error(`Error: No source directory specified (see the --source CLI argument)`);
	if(!fs.existsSync(settings.source))
		throw new Error(`Error: The source directory at '${settings.source}' doesn't exist or you haven't got permission to access it.`);
	
	
	const uniq_manager = new RecordUniqManager(settings.count_file);
	await uniq_manager.deduplicate(settings.source, settings.target);
}