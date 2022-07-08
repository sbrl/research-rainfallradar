"use strict";

import fs from 'fs';
import path from 'path';
import { Readable } from 'stream';

import nexline from 'nexline';
import pretty_ms from 'pretty-ms';
import gunzip from 'gunzip-maybe';

import RecordsWriter from './RecordsWriter.mjs';

async function records_recompress(dirpath_source, dirpath_target, items_per_file) {
	const files = (await fs.promises.readdir(dirpath_source)).map(filename => path.join(dirpath_source, filename));
	
	const reader = nexline({
		input: files.map(filepath => new Readable().wrap(fs.createReadStream(filepath).pipe(gunzip())))
	});
	
	
	if(!fs.existsSync(dirpath_target))
		await fs.promises.mkdir(dirpath_target, { recursive: true });
	
	let writer = null, i = 0, i_file = 0, i_this_file;
	let time_start = new Date(), time_display = time_start;
	for await(const line of reader) {
		if(line === "") continue;
		
		if(writer === null || i_this_file >= items_per_file) {
			if(writer !== null) await writer.close();
			writer = new RecordsWriter(path.join(dirpath_target, `${i_file}.jsonl.gz`));
			i_file++; i_this_file = 0;
		}
		
		await writer.write_raw(line.trim());
		
		i++;
		i_this_file++;
		
		if(new Date() - time_display > 2000) {
			const elapsed = new Date() - time_start;
			process.stdout.write(`${pretty_ms(elapsed, { keepDecimalsOnWholeSeconds: true })} elapsed | ${i_file}/${i_this_file}/${i} files/this file/total |  ${(1000 / (elapsed / i)).toFixed(2)} lines/sec | ${items_per_file - i_this_file} left for this file\r`)
		}
	}
	await writer.close();
	
	return { recompress_lines: i, recompress_files: i_file };
}

export default records_recompress;