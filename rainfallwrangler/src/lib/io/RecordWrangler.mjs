"use strict";

import fs from 'fs';
import path from 'path';

import log from '../../lib/io/NamespacedLog.mjs'; const l = log("recordwrangler");

import RecordBuilder from '../record/RecordBuilder.mjs';
import RecordsWriter from '../record/RecordsWriter.mjs';
import pretty_ms from 'pretty-ms';
import { end_safe } from './StreamHelpers.mjs';

class RecordWrangler {
	#builder = new RecordBuilder();
	
	constructor(dirpath, count_per_file) {
		this.dirpath = dirpath;
		this.count_per_file = count_per_file;
		
		this.display_interval = 2 * 1000;
		
		if(!fs.existsSync(this.dirpath))
			fs.mkdirSync(this.dirpath);
	}
	
	async write(reader_radar, reader_water) {
		// TODO: Shuffle stuff about in the *Python* data pipeline
		
		let writer = null;
		let i = 0, i_file = 0, count_this_file = 0, time_start = new Date(), time_display = time_start;
		while(true) {
			i++;
			
			// Start writing to a new file when necessary
			if(writer == null || count_this_file > this.count_per_file) {
				if(writer !== null) await writer.close();
				const filepath_next = path.join(this.dirpath, `${i_file}.jsonl.gz`);
				writer = new RecordsWriter(filepath_next);
				i_file++;
				count_this_file = 0;
			}
			
			count_this_file++;
			
			const sample_radar = await reader_radar.next();
			const sample_water = await reader_water.next();
			if(sample_radar.done || sample_water.done) {
				l.log(`Done because ${sample_radar.done?"radar":"water"} reader is out of records`);
				break;
			}
			
			const example_next = this.make_example(
				sample_radar.value,
				sample_water.value
			);
			
			await writer.write(example_next);
			
			const time_now = new Date();
			if(time_now - time_display > this.display_interval) {
				const elapsed = new Date() - time_start;
				process.stderr.write(`Elapsed: ${pretty_ms(elapsed, { keepDecimalsOnWholeSeconds: true })}, Written ${count_this_file}/${i_file}/${i} examples/files/total | ${(1000 / (elapsed / i)).toFixed(2)} batches/sec | ${this.count_per_file - count_this_file} left for this file\r`);
				time_display = time_now;
			}
		}
		await writer.close();
		
		console.log(`\nComplete! ${i_file}/${i} files/records_total written in ${pretty_ms(new Date() - time_start)}\n`);
	}
	
	make_example(sample_radar, sample_water) {
		this.#builder.add("rainfallradar", sample_radar);
		this.#builder.add("waterdepth", sample_water);
		return this.#builder.release();
	}
}

export default RecordWrangler;
