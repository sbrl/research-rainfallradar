"use strict";

import fs from 'fs';
import { Readable } from 'stream';

import nexline from 'nexline';
import gunzip from 'gunzip-maybe';

/**
 * Reads the records from a (potentially gzipped) .jsonl / .jsonl.gz file.
 * @param	{string}					filename	The filename to read from.
 * @return	{AsyncGenerator<string>}	An asynchronous generator that iteratively returns the lines in the file.
 */
function records_read(filename) {
	return nexline({
		input: new Readable().wrap(fs.createReadStream(filename).pipe(gunzip()))
	});
}

export default records_read;