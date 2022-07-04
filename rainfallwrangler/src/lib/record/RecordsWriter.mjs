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
	
	/**
	 * Writes a sample to the file, followed by a new line \n character.
	 * @param	{Map}		sample	The sample to write.
	 * @return	{Promise}
	 */
	async write(sample) {
		const str = JSON.stringify(Object.fromEntries(sample));
		await write_safe(this.#gzip, str+"\n");
	}
	
	/**
	 * Writes a raw value to the file, followed by a new line \n character.
	 * @param	{string}	line	The thing to write.
	 * @return	{Promise}
	 */
	async write_raw(line) {
		await write_safe(this.#gzip, line+"\n");
	}
	
	/**
	 * Closes the underlying file gracefully.
	 * No more may be written to the file after this method is called.
	 * @return	{Promise}
	 */
	async close() {
		await end_safe(this.#gzip);
		await end_safe(this.#stream_out);
	}
}

export default RecordsWriter;