"use strict";

import fs from 'fs';
import child_process from 'child_process';

import nexline from nexline;

import log from './NamespacedLog.mjs'; const l = log("gzipchildprocess");
import { end_safe } from './StreamHelpers.mjs';
import { fstat } from 'fs';


const __dirname = import.meta.url.slice(7, import.meta.url.lastIndexOf("/"));

async function* py_jsonl2tfrecord(filepath_source, filepath_target, filepath_meta=null) {
	// get stdin() { return this.child_process.stdin; }
	// get stdout() { return this.child_process.stdout; }
	// get stderr() { return this.child_process.stderr; }
	
	
	child_process = child_process.spawn(
		"python3", [
			path.join(__dirname, "json2tfrecord.py"),
			"--input", filepath_source,
			"--output", filepath_target
		], { // TODO: detect binary - python3 vs python
			// Pipe stdin + stdout; send error to the parent process
			stdio: [ "ignore", "pipe", "inherit" ]
		}
	);
	
	const reader = nexline({ input: child_process.stdout });
	
	for await(const line of reader) {
		if(line.startsWith("SHAPE") && filepath_meta !== null ) {
			await fs.promises.writeFile(
				filepath_meta,
				line.split(/\t+/)[1]
			);
			continue;
		}
		
		yield parseInt(line, 10);
	}
}

export default py_jsonl2tfrecord;
