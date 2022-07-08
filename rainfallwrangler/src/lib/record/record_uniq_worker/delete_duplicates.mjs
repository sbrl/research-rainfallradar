"use strict";

import fs from 'fs';

import records_read from "../records_read.mjs";
import RecordsWriter from '../RecordsWriter.mjs';

import log from '../../io/NamespacedLog.mjs'; const l = log("recorduniq:worker");

// This could be muxed together rather than use a worker like this in the main thread since it's I/O bound
export default async function(filepath_source, lines) {
	
	l.info(`DEBUG lines slated for deletion`, lines);
	const filepath_tmp = `${filepath_source}.dupedeleteTMP`;
	let i = -1, count_deleted = 0, writer = new RecordsWriter(filepath_tmp);
	for await(const line of records_read(filepath_source)) {
		i++;
		if(line === "") continue;
		if(lines.includes(i)) {
			count_deleted++;
			continue;
		}
		
		await writer.write_raw(line);
	}
	await writer.close();
	
	await fs.promises.rename(filepath_tmp, filepath_source);
	
	l.log(`Deleted`, count_deleted, `lines out of`, lines.length, `slated`);
	
	return count_deleted;
}
