"use strict";

import fs from 'fs';
import path from 'path';

import tfrecord from 'tfrecord-stream';

class TFRecordWriter {
	constructor(dirpath, count_per_file) {
		this.dirpath = dirpath;
		this.count_per_file = count_per_file;
		
		if(!fs.existsSync(dirpath))
			fs.mkdirSync(dirpath);
		
		this.#builder = tfrecord.createBuilder();
	}
	
	write(reader_radar, reader_water) {
		// TODO: Shuffle stuff about in the *Python* data pipeline
		
		let writer = null;
		let i = -1, i_file = 0, count_this_file = 0;
		while(true) {
			i++;
			
			// Start writing to a new file when necessary
			if(writer == null || count_this_file > this.count_per_file) {
				if(writer !== null) await writer.close();
				const filepath_next = path.join(dirpath, `${i_file}.tfrecord`);
				writer = await tfrecord.Writer.createFromStream(
					fs.createWriteStream(filepath_next)
				);
				i_file++;
			}
			
			const sample_radar = await reader_radar.next();
			const sample_water = await reader_water.next();
			
			if(sample_radar.done || sample_water.done) break;
			
			const example_next = this.make_example(
				sample_radar.value,
				sample_water.value
			);
			
			await writer.writeExample(example_next);
		}
		
	}
	
	make_example(sample_radar, sample_water) {
		this.#builder.setFloats("rainfallradar", sample_radar.flat());
		this.#builder.setInteger("rainfallradar_width", sample_radar[0].length);
		this.#builder.setFloats("waterdepth", sample_water.flat());
		this.#builder.setInteger("waterdepth_width", sample_water[0].length);
		
		return this.#builder.releaseExample();
	}
}

export default TFRecordWriter;
