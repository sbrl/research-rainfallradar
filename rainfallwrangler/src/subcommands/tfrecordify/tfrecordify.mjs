"use strict";

import settings from '../../settings.mjs';

export default async function() {
	if(typeof settings.water !== "string")
		throw new Error(`Error: No filepath to water depth data specified.`);
	
	if(typeof settings.rainfall !== "string")
		throw new Error(`Error: No filepath to rainfall radar data specified.`);
	
	
	// TODO: Do the fanceh parsing stuff here
}
