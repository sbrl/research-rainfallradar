"use strict";

class RecordBuilder {
	constructor() {
		this.acc = new Map();
	}
	
	add(key, value) {
		this.acc.set(key, value);
	}
	
	release() {
		const result = this.acc;
		this.acc = new Map();
		return result;
	}
}

export default RecordBuilder;