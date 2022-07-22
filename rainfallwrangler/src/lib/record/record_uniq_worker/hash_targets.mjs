"use strict";

import crypto from 'crypto';

import records_read from "../records_read.mjs";

import log from '../../io/NamespacedLog.mjs'; const l = log("recorduniq:worker:hash");

export default async function(filepath) {
	const result = [];
	
	let i = -1;
	for await(const line of records_read(filepath)) {
		i++;
		if(line === "") continue;
		
		// Ref https://stackoverflow.com/a/58307338/1460422
		result.push({ i, hash: crypto.createHash("sha256").update(line, "binary").digest("base64") });
	}
	
	l.log(`${filepath}: Hashed ${i+1} lines`);
	
	return result;
}