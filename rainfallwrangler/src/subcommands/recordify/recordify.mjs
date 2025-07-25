"use strict";

import fs from 'fs';

import settings from '../../settings.mjs';
import RecordWrangler from '../../lib/io/RecordWrangler.mjs';
import RadarWrangler from '../../lib/RadarWrangler.mjs';
import Terrain50StreamReader from '../../lib/io/Terrain50StreamReader.mjs';

import log from '../../lib/io/NamespacedLog.mjs'; const l = log("recordify");

export default async function() {
	if(typeof settings.water !== "string")
		throw new Error(`Error: No filepath to water depth data specified.`);
	
	if(typeof settings.rainfall !== "string")
		throw new Error(`Error: No filepath to rainfall radar data specified.`);
	
	if(typeof settings.output !== "string")
		throw new Error(`Error: No output directory specified.`);
	if(typeof settings.count_file !== "number")
		throw new Error(`Error: --count-file was not specified.`);
	if(isNaN(settings.count_file))
		throw new Error(`Error: --count-file was not a number. process.argv: ${process.argv.join(" ")}`);
	
	if(!fs.existsSync(settings.output))
		await fs.promises.mkdir(settings.output, { recursive: true });
	
	console.log("DEBUG", settings);
	
	// Recordify CLEAN, does NOT shuffle
	const writer = new RecordWrangler(settings.output, settings.count_file);
	const reader_radar = new RadarWrangler(settings.rainfall_pattern);
	const reader_water = new Terrain50StreamReader(settings.threshold);
	
	await writer.write(reader_radar.iterate(settings.rainfall), reader_water.iterate(settings.water));
	
	l.log("Closing radar reader")
	await reader_radar.close();
	l.log("Closing water depth data reader")
	await reader_water.close();
	l.log(`All streams closed`);
}
