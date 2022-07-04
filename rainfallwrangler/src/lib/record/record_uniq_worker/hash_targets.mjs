"use strict";

import crypto from 'crypto';

import records_read from "../records_read.mjs";

import log from '../io/NamespacedLog.mjs'; const l = log("recorduniq:worker");

export default async function(filename) {
	const result = [];
	
	let i = -1;
	for await(const line of records_read(filename)) {
		i++;
		if(line === "") continue;
		
		// Ref https://stackoverflow.com/a/58307338/1460422
		result.push({ i, hash: crypto.createHash("sha256").update(line, "binary").digest("base64") });
	}
	
	return result;
}