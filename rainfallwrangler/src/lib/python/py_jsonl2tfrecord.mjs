"use strict";

import fs from 'fs';
import path from 'path';
import child_process from 'child_process';
import { Readable } from 'stream';

import nexline from 'nexline';

import log from '../io/NamespacedLog.mjs'; const l = log("gzipchildprocess");
// import { end_safe } from '../io/StreamHelpers.mjs';

function snore(ms) {
	return new Promise((resolve, _reject) => setTimeout(resolve, ms));
}

const __dirname = import.meta.url.slice(7, import.meta.url.lastIndexOf("/"));

async function* py_jsonl2tfrecord(filepath_source, filepath_target, filepath_meta=null) {
	// get stdin() { return this.converter.stdin; }
	// get stdout() { return this.converter.stdout; }
	// get stderr() { return this.converter.stderr; }
	
	const env = {}; Object.assign(env, process.env);
	if(filepath_meta !== null) env["NO_SILENCE"] = "NO_SILENCE";
	
	const converter = child_process.spawn(
		"python3", [
			path.join(__dirname, "json2tfrecord.py"),
			"--input", filepath_source,
			"--output", filepath_target
		], { // TODO: detect binary - python3 vs python
			// Pipe stdin + stdout; send error to the parent process
			stdio: [ "ignore", "pipe", "inherit" ],
			env
		}
	);
	// converter.stdout.on("data", (chunk) => console.log(`DEBUG chunk`, chunk));
	
	const reader = nexline({ input: new Readable().wrap(converter.stdout) });
	
	for await(const line of reader) {
		if(line.startsWith("SHAPES\t")) {
			if(filepath_meta !== null) {
				await fs.promises.writeFile(
					filepath_meta,
					line.split(/\t+/)[1]
				);
			}
			continue;
		}
		yield parseInt(line, 10);
	}
}

export default py_jsonl2tfrecord;
