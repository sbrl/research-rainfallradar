"use strict";

class RadarWrangler {
	/**
	 * The total number of timesteps we need in the buffer before we can even consider generating a sample.
	 * @return {number}
	 */
	get timesteps_required() {
		return this.channel_pattern.reduce((next, total) => total + next, 0);
	}
	
	constructor(channel_pattern, pooling_operator="max", time_step_interval = 300) {
		this.channel_pattern = channel_pattern;
		this.pooling_operator = pooling_operator;
		
		this.reader = new RadarReader(time_step_interval);
	}
	
	async *iterate(filepath) {
		const timestep_buffer = [];
		for await(const next of this.reader.iterate(filepath)) {
			timestep_buffer.push(next.data);
			
			const result = this.make_sample(timestep_buffer);
			if(result == null) continue;
			
			yield result;
			
			timestep_buffer.shift();
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

export default RadarWrangler;
