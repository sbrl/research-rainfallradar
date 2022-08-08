"use strict";

import fs from 'fs';
import path from 'path';
import os from 'os';
import p_map from 'p-map';
import pretty_ms from 'pretty-ms';

import debounce from '../async/debounce.mjs';
import py_jsonl2tfrecord from '../python/py_jsonl2tfrecord.mjs';
import log from '../../lib/io/NamespacedLog.mjs'; const l = log("jsonl2tf");


/**
 * Converts a directory of .jsonl.gz files to .tfrecord.gz files.
 * @param	{string}	dirpath_source	The source directory to read from.
 * @param	{string}	dirpath_target	The target directory to write to.
 * @return	{void}
 */
export default async function(dirpath_source, dirpath_target) {
	const files = await fs.promises.readdir(dirpath_source);
	
	let time_start = new Date(), lines_processed = 0, files_complete = 0;
	
	const update_progress_force = () => {
		process.stdout.write(`${files_complete}/${lines_processed} files/lines complete | ${((new Date() - time_start) / lines_processed).toFixed(3)} lines/sec | ${((files_processed / files.length)*100).toFixed(2)}% complete\r`);
	};
	const update_progress = debounce(update_progress_force);
	
	await p_map(files, async (filename, i) => {
		const filepath_source = path.join(dirpath_source, filename);
		const filepath_dest = path.join(dirpath_target, filename.replace(/\.jsonl\.gz$/, ".tfrecord.gz"));
		const filepath_meta = i === 0 ? path.join(dirpath_target, `metadata.json`) : null;
		l.log(`start ${i} | ${filename} | META ${filepath_meta}`);
		let time_start = new Date(), lines_done = 0;
		for await (let line_number of py_jsonl2tfrecord(filepath_source, filepath_dest, filepath_meta)) {
			lines_processed++;
			lines_done = line_number;
			update_progress();
		}
		files_complete++;
		l.log(`converted ${filename}: ${lines_done} lines @ ${pretty_ms((new Date() - time_start) / lines_done)}`);
	}, { concurrency: os.cpus().length });
	update_progress_force();
	l.log(`complete: ${lines_processed}/${files_complete} lines/files processed in ${pretty_ms(new Date() - time_start)}`);
}
