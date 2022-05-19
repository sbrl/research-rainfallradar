"use strict";

import { pipeline } from 'stream';
import util from 'util';
import fs from 'fs';
import path from 'path';


import log from './lib/io/NamespacedLog.mjs'; const l = log("reader:terrain50stream");

import array2d_classify_convert_bin from '../../manip/array2d_classify_convert_bin.mjs';

class Terrain50StreamReader {	
	constructor(threshold = 0.1, tolerant = false) {
		this.threshold = threshold;
		
		this.tolerant = tolerant;
	}
	
	async *iterate(filepath) {
		const stream = Terrain50.ParseStream(pipeline(
			fs.createReadStream(filepath),
			gunzip()
		), this.tolerant ? /\s+/ : " ");
		
		let i = -1;
		for await (const next of stream) {
			i++;
			// Skip the first few items, because we want to predict the water
			// depth after the rainfall radar data
			if(i < this.offset)
				continue;
			
			const values_bin = array2d_classify_convert_bin(
				next.data,
				this.threshold
			);
			
			// l.debug(`[DEBUG:Terrain50Stream] values_bin`, util.inspect(values_bin).substr(0, 500));
			
			// l.debug(`[Terrain50Stream] Yielding tensor of shape`, values_bin.shape);
			yield values_bin;
		}
	}
}

export default Terrain50StreamReader;
