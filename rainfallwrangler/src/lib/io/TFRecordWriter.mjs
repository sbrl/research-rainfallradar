"use strict";

import fs from 'fs';
import path from 'path';

import tfrecord from 'tfrecord-stream';
import pretty_ms from 'pretty-ms';

class TFRecordWriter {
	#builder = tfrecord.createBuilder();
	
	constructor(dirpath, count_per_file) {
		this.dirpath = dirpath;
		this.count_per_file = count_per_file;
		
		if(!fs.existsSync(this.dirpath))
			fs.mkdirSync(this.dirpath);
	}
	
	async write(reader_radar, reader_water) {
		// TODO: Shuffle stuff about in the *Python* data pipeline
		
		let writer = null;
		let i = -1, i_file = 0, count_this_file = 0, time_start = new Date();
		while(true) {
			i++;
			
			// Start writing to a new file when necessary
			if(writer == null || count_this_file > this.count_per_file) {
				if(writer !== null) await writer.close();
				const filepath_next = path.join(this.dirpath, `${i_file}.tfrecord`);
				writer = await tfrecord.Writer.createFromStream(
					fs.createWriteStream(filepath_next)
				);
				i_file++;
			}
			
			const sample_radar = await reader_radar.next();
			console.log(`SAMPLE_RADAR`);
			const sample_water = await reader_water.next();
			console.log(`SAMPLE_WATER`);
			
			if(sample_radar.done || sample_water.done) break;
			
			const example_next = this.make_example(
				sample_radar.value,
				sample_water.value
			);
			
			await writer.writeExample(example_next);
			
			process.stderr.write(`Elapsed: ${pretty_ms(new Date() - time_start)}, Written ${count_this_file}/${i_file}/${i} examples/files/total\r`);
		}
		
	}
	
	make_example(sample_radar, sample_water) {
		console.log(`SAMPLE WATER ${sample_water.flat().length} RAINFALL ${sample_radar.flat().length}`);
		const sample_radar_flat1 = sample_radar.flat();
		this.#builder.setFloats("rainfallradar", sample_radar_flat1.flat());
		this.#builder.setInteger("rainfallradar_width", sample_radar[0].length);
		this.#builder.setInteger("rainfallradar_channelsize", sample_radar_flat1[0].length);
		this.#builder.setFloats("waterdepth", sample_water.flat());
		this.#builder.setInteger("waterdepth_width", sample_water[0].length);
		
		return this.#builder.releaseExample();
	}
}

export default TFRecordWriter;