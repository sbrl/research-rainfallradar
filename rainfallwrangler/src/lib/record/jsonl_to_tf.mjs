"use strict";

import fs from 'fs';
import path from 'path';
import p_map from 'p-map';
import pretty_ms from 'pretty-ms';

import debounce from '../async/debounce.mjs';
import py_jsonl2tfrecord from '../python/py_jsonl2tfrecord.mjs';
import log from '../../lib/io/NamespacedLog.mjs'; const l = log("jsonl2tf");

export default async function(dirpath_source, dirpath_target) {
	const files = fs.promises.readdir(dirpath_source);
	
	let time_start = new Date(), lines_processed = 0, files_complete = 0;
	
	const update_progress = debounce(() => {
		process.stdout.write(`${files_complete}/${lines_processed} files/lines complete | ${((new Date() - time_start) / lines_processed).toFixed(3)} lines/sec | ${((files_processed / files.length)*100).toFixed(2)}% complete\r`);
	});
	
	p_map(files, async (filename, i) => {
		const filepath_source = path.join(dirpath_source, filename);
		const filepath_dest = path.join(dirpath_target, filename);
		const filepath_meta = i === 0 ? path.join(dirpath_target, `metadata.json`) : null;
		let time_start = new Date(), lines_done = 0;
		for await (let line_number of py_jsonl2tfrecord(filepath_source, filepath_dest, filepath_meta)) {
			lines_processed++;
			lines_done = line_number;
			update_progress();
		}
		files_complete++;
		l.log(`converted ${filename}: ${lines_done} lines @ ${pretty_ms((new Date() - time_start) / lines_done)}`);
	});
}
