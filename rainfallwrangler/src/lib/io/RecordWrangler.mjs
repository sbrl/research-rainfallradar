"use strict";

import fs from 'fs';
import path from 'path';

import RecordBuilder from '../record/RecordBuilder.mjs';
import RecordsWriter from '../record/RecordsWriter.mjs';
import pretty_ms from 'pretty-ms';

class RecordWrangler {
	#builder = new RecordBuilder();
	
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
			
			console.log(`RecordWriter step ${i}`);
			
			// Start writing to a new file when necessary
			if(writer == null || count_this_file > this.count_per_file) {
				if(writer !== null) await writer.close();
				const filepath_next = path.join(this.dirpath, `${i_file}.jsonl.gz`);
				writer = new RecordsWriter(filepath_next);
				console.log(`RecordWriter NEW FILE ${filepath_next}`);
				i_file++;
			}
			
			const sample_radar = await reader_radar.next();
			const sample_water = await reader_water.next();
			
			if(sample_radar.done || sample_water.done) break;
			
			const example_next = this.make_example(
				sample_radar.value,
				sample_water.value
			);
			
			await writer.write(example_next);
			
			process.stderr.write(`Elapsed: ${pretty_ms(new Date() - time_start)}, Written ${count_this_file}/${i_file}/${i} examples/files/total\r`);
		}
		
	}
	
	make_example(sample_radar, sample_water) {
		this.#builder.add("rainfallradar", sample_radar);
		this.#builder.add("waterdepth", sample_water.flat);
		return this.#builder.release();
	}
}

export default RecordWrangler;
