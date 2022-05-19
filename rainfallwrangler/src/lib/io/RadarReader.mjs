"use strict";

import fs from 'fs';
import { Readable } from 'stream';

import nexline from 'nexline';
import gunzip from 'gunzip-maybe';

import log from './NamespacedLog.mjs'; const l = log("reader:radar");
import interpolate from '../manip/array2d_interpolate.mjs';
import transpose from '../manip/array2d_transpose.mjs';

/**
 * Reads data in order from a directory of .jsonl.gz files.
 * @param	{string}	in_stride		Return only every X objects. 1 = return every object. Default: 1
 * @param	{boolean}	do_interpolate	Whether to interpolate to fill in missing values.
 * @param	{number}	time_step_interval	The nominal interval, in seconds, between time steps (default: 300 seconds)
 */
class RadarReader {
	constructor(time_step_interval = 300, in_stride = 1, do_interpolate = true) {
		this.time_step_interval = time_step_interval;
		this.stride = in_stride;
		this.do_interpolate = do_interpolate;
		
		// this.writer_interp_stats = [];
	}
	
	/**
	 * An async iterator that yields rainfall radar objects in order.
	 * Note that for a single RadarReader object, this method may be called
	 * multiple times - potentially in parallel.
	 * @param	{string}				filename	The filename to read from.
	 * @return	{Generator<Promise>}	The async generator.
	 */
	async *iterate(filename) {
		if(!fs.existsSync(filename))
			throw new Error(`RadarReader/Error: Can't read from '${filename}' as it doesn't exist.`);
		
		let read = fs.createReadStream(filename),
			extractor = gunzip();
		read.pipe(extractor);
		
		let reader = nexline({
			input: new Readable().wrap(extractor) // Wrap the stream so that nexline likes it
		});
		
		let i = -1;
		let prev = null
		while(true) {
			let next_line = await reader.next();
			if(next_line == null)
				break;
			
			i++;
			if(i % this.stride !== 0) continue;
			
			// Ignore empty lines
			if(next_line.trim() === "")
				continue;
			
			let next = null;
			try {
				next = JSON.parse(next_line);
			} catch(error) {
				l.warn(`Encountered invalid JSON object at line ${i}, skipping (error: ${error.message})`);
				continue;
			}
			
			if(next == null) continue;
			
			// Sort out the timestamp
			if(next.timestamp == null) {
				if(next.timestamps !== null) next.timestamp = next.timestamps[1];
				else {
					l.warn(`Encountered JSON object without a timestamp`);
					continue;
				}
			}
			next.timestamp = new Date(next.timestamp);
			// Transpose the data to correct our earlier mistake
			next.data = transpose(next.data) // Correct the orientation of the data array
			let tmp = next.size_extract.height;
			next.size_extract.height = next.size_extract.width;
			next.size_extract.height = tmp;
			// Now, the data is in the format data[y][x]
			
			// Interpolate if needed
			if(this.do_interpolate && prev !== null && next.timestamp - prev.timestamp > this.time_step_interval * 1000 * 1.5 * this.stride) {
				for await(let item of this.interpolate(prev, next)) {
					yield item;
				}
			}
			
			// We've caught up (if required) - continue as normal
			yield next;
			prev = next;
		}
	}
	
	/**
	 * Interpolates between 2 objects.
	 * @param	{Object}	a	The first object.
	 * @param	{Object}	b	The second object.
	 * @return	{Generator<Object>}
	 */
	async *interpolate(a, b) {
		let next_timestamp = new Date(a.timestamp); // This clones the existing Date object
		
		// Increment the time interval
		next_timestamp.setSeconds(
			next_timestamp.getSeconds() + (this.time_step_interval * this.stride)
		);
		
		do {
			// The percentage of the way through the interpolation we are
			let interpolation_percentage = 1 - ((b.timestamp - next_timestamp) / (b.timestamp - a.timestamp));
			
			// Generate a temporary interpolated object
			let obj_interpolated = {};
			Object.assign(obj_interpolated, a);
			obj_interpolated.timestamp = next_timestamp;
			obj_interpolated.data = interpolate(
				a.data, b.data,
				interpolation_percentage
			);
			
			// this.writer_interp_stats.push(next_timestamp);
			
			yield obj_interpolated;
			
			// Increment the time interval
			next_timestamp.setSeconds(
				next_timestamp.getSeconds() + (this.time_step_interval * this.stride)
			);
		} while(b.timestamp - next_timestamp >= this.time_step_interval * 1000 * this.stride);
	}
}

export default RadarReader;
