"use strict";

import fs from 'fs';

import SpawnStream from 'spawn-stream';

import { write_safe, end_safe } from '../io/StreamHelpers.mjs';

class RecordsWriter {
	#stream_out = fs.createWriteStream(filepath);
	#gzip = SpawnStream("gzip");
	
	constructor(filepath) {
		this.#gzip.pipe(this.#stream_out);
	}
	
	async write(sample) {
		console.log(sample);
		await write_safe(this.#gzip, JSON.stringify(sample));
	}
	
	async close() {
		await this.#gzip.close();
		await this.#stream_out.close();
	}
}

export default RecordsWriter;