"use strict";

import fs from 'fs';

import log from '../../lib/io/NamespacedLog.mjs'; const l = log("recordswriter");

import GzipChildProcess from '../io/GzipChildProcess.mjs';
import { write_safe, end_safe } from '../io/StreamHelpers.mjs';

class RecordsWriter {
	#stream_out = null;
	#gzip = new GzipChildProcess();
	
	constructor(filepath) {
		this.#stream_out = fs.createWriteStream(filepath);
		this.#gzip.stdout.pipe(this.#stream_out);
	}
	
	/**
	 * Writes a sample to the file, followed by a new line \n character.
	 * @param	{Map}		sample	The sample to write.
	 * @return	{Promise}
	 */
	async write(sample) {
		const str = JSON.stringify(Object.fromEntries(sample));
		await write_safe(this.#gzip.stdin, str+"\n");
	}
	
	/**
	 * Writes a raw value to the file, followed by a new line \n character.
	 * @param	{string}	line	The thing to write.
	 * @return	{Promise}
	 */
	async write_raw(line) {
		await write_safe(this.#gzip.stdin, line+"\n");
	}
	
	/**
	 * Closes the underlying file gracefully.
	 * No more may be written to the file after this method is called.
	 * @return	{Promise}
	 */
	async close() {
		await this.#gzip.close();
		// Closing this.#stream_out causes a silent crash O.o 2022-07-08 @sbrl Node.js 18.4.0
	}
}

export default RecordsWriter;