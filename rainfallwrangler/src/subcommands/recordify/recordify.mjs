"use strict";

import fs from 'fs';

import settings from '../../settings.mjs';
import RecordWrangler from '../../lib/io/RecordWrangler.mjs';
import RadarWrangler from '../../lib/RadarWrangler.mjs';
import Terrain50StreamReader from '../../lib/io/Terrain50StreamReader.mjs';

import log from './NamespacedLog.mjs'; const l = log("recordify");

export default async function() {
	if(typeof settings.water !== "string")
		throw new Error(`Error: No filepath to water depth data specified.`);
	
	if(typeof settings.rainfall !== "string")
		throw new Error(`Error: No filepath to rainfall radar data specified.`);
	
	if(typeof settings.output !== "string")
		throw new Error(`Error: No output directory specified.`);
	
	if(!fs.existsSync(settings.output))
		await fs.promises.mkdir(settings.output, { recursive: true });
	
	console.log("DEBUG", settings);
	const writer = new RecordWrangler(settings.output, settings.count_file);
	const reader_radar = new RadarWrangler(settings.rainfall_pattern);
	const reader_water = new Terrain50StreamReader();
	
	await writer.write(reader_radar.iterate(settings.rainfall), reader_water.iterate(settings.water));
	
	l.log("Closing reader reader")
	await reader_radar.close();
	l.log("Closing water depth data reader")
	await reader_water.close();
	l.log(`All streams closed`);
}
