"use strict";

import { pipeline } from 'stream';
import util from 'util';
import fs from 'fs';
import path from 'path';


import log from './lib/io/NamespacedLog.mjs'; const l = log("reader:terrain50stream");

import array2d_classify_convert_bin from '../../manip/array2d_classify_convert_bin.mjs';

class Terrain50StreamReader {
	/**
	 * The tital number of timesteps we need in the buffer before we can even consider generating a sample.
	 * @return {number}
	 */
	get timesteps_required() {
		return this.channel_pattern.reduce((next, total) => total + next, 0);
	}
	
	constructor(threshold, channel_pattern, pooling_operator="max", tolerant = false) {
		this.threshold = threshold;
		this.channel_pattern = channel_pattern;
		
		this.pooling_operator = "max";
		this.tolerant = tolerant;
	}
	
	async *iterate(filepath) {
		const stream = Terrain50.ParseStream(pipeline(
			fs.createReadStream(filepath),
			gunzip()
		), this.tolerant ? /\s+/ : " ");
		
		const timestep_buffer = [];
		
		let i = -1;
		for await (const next of stream) {
			i++;
			// Skip the first few items, because we want to predict the next
			// timestep after the rainfall radar data
			if(i < this.temporal_depth)
				continue;
			
			const values_bin = array2d_classify_convert_bin(
				next.data,
				this.threshold
			);
			
			timestep_buffer.push(values_bin);
			// l.debug(`[DEBUG:Terrain50Stream] values_bin`, util.inspect(values_bin).substr(0, 500));
			
			const result = this.make_sample(timestep_buffer);
			if(result == null) continue;
			// l.debug(`[Terrain50Stream] Yielding tensor of shape`, values_bin.shape);
			yield result;
		}
		
		yield this.make_sample(timestep_buffer);
	}
	
	make_sample(timestep_buffer) {
		if(timestep_buffer.length < this.timesteps_required) return null;
		
		const grouped_timesteps = [];
		let offset = 0;
		for(const channel_timestep_count of this.channel_pattern) {
			const acc = [];
			for(let i = offset; i < channel_timestep_count+offset; i++) {
				acc.push(timestep_buffer[i]);
			}
			
			grouped_timesteps.push(array2d_pool(acc, this.pooling_operator));
			offset += channel_timestep_count;
		}
	}
}

export default Terrain50StreamReader;
