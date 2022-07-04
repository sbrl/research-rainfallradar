"use strict";

import crypto from 'crypto';

import records_read from "../records_read.mjs";
import RecordsWriter from '../RecordsWriter.mjs';

import log from '../io/NamespacedLog.mjs'; const l = log("recorduniq:worker");

// This could be muxed together rather than use a worker like this in the main thread since it's I/O bound
export default async function(filepath, lines) {
	const result = [];
	
	let i = -1, writer = new RecordsWriter(filepath);
	for await(const line of records_read(filename)) {
		i++;
		if(line === "" || lines.includes(i)) continue;
		
		await writer.write_raw(line);
	}
	await writer.close();
	
	return result;
}
