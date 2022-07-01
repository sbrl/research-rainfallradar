"use strict";

import fs from 'fs';

import ChildProcess from 'duplex-child-process';

import { write_safe, end_safe } from '../io/StreamHelpers.mjs';

class RecordsWriter {
	#stream_out = null;
	#gzip = ChildProcess.spawn("gzip");
	
	constructor(filepath) {
		this.#stream_out = fs.createWriteStream(filepath);
		this.#gzip.pipe(this.#stream_out);
	}
	
	async write(sample) {
		const str = JSON.stringify(Object.fromEntries(sample));
		await write_safe(this.#gzip, str);
	}
	
	async close() {
		await end_safe(this.#gzip);
		await end_safe(this.#stream_out);
	}
}

export default RecordsWriter;