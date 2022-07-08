"use strict";

import util from 'util';
import { Readable } from 'stream';
import fs from 'fs';
import path from 'path';

import gunzip from 'gunzip-maybe';
import Terrain50 from 'terrain50';

import log from './NamespacedLog.mjs'; const l = log("reader:terrain50stream");
import array2d_classify_convert_bin from '../manip/array2d_classify_convert_bin.mjs';
import { end_safe }

class Terrain50StreamReader {	
	constructor(threshold = 0.1, tolerant = false) {
		this.threshold = threshold;
		
		this.tolerant = tolerant;
		
		this.stream_in = null;
		this.stream_extractor = null;
	}
	
	async *iterate(filepath) {
		this.stream_in = fs.createReadStream(filepath);
		this.stream_extractor = gunzip();
		
		const stream = Terrain50.ParseStream(
			new Readable().wrap(this.stream_in.pipe(this.stream_extractor)),
			this.tolerant ? /\s+/ : " "
		);
		
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
	
	async close() {
		if(this.stream_in !== null) this.stream_in.close();
		if(this.stream_extractor !== null) await end_safe(this.stream_extractor);
		
		this.stream_in = null
		this.stream_extractor = null;
	}
}

export default Terrain50StreamReader;
