"use strict";

import fs from 'fs';

import settings from '../../settings.mjs';
import TFRecordWriter from '../../lib/io/TFRecordWriter.mjs';
import RadarReader from '../../lib/io/RadarReader.mjs';
import Terrain50StreamReader from '../../lib/io/Terrain50StreamReader.mjs';


export default async function() {
	if(typeof settings.water !== "string")
		throw new Error(`Error: No filepath to water depth data specified.`);
	
	if(typeof settings.rainfall !== "string")
		throw new Error(`Error: No filepath to rainfall radar data specified.`);
	
	if(typeof settings.output !== "string")
		throw new Error(`Error: No output directory specified.`);
	
	if(!fs.existsSync(settings.output))
		await fs.promises.mkdir(settings.output, { recursive: true });
		
	const writer = new TFRecordWriter(settings.output, settings.count_file);
	const reader_radar = new RadarReader();
	const reader_water = new Terrain50StreamReader();
	
	await writer.write(reader_radar, reader_water);
}
