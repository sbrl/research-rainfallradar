"use strict";

import fs from 'fs';

import settings from "../../settings.mjs";

import RecordUniqManager from '../../lib/record/RecordUniqManager.mjs';

export default async function() {
	if(typeof settings.cli.source !== "string")
		throw new Error(`Error: No source directory specified (see the --source CLI argument)`);
	if(!fs.existsSync(settings.cli.source))
		throw new Error(`Error: The source directory at '${settings.cli.source}' doesn't exist or you haven't got permission to access it.`);
	
	
	const uniq_manager = new RecordUniqManager(settings.cli.count_file);
	await uniq_manager.deduplicate(settings.cli.source, settings.cli.target);
}