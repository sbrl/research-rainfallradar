"use strict";

/**
 * Returns a function that debounces the given function.
 * The function will only be called every INTERVAL number of milliseconds - additional
 * calls return immediately without calling the provided function.
 * Useful e.g. debouncing the scroll event in a browser.
 * Note that p-debounce throttles the function, whereas this debounce implementation CANCELS additional calls to it.
 * @param	{function}	fn			The function to debounce.
 * @param	{number}	interval	The interval - in milliseconds - that the function should be called at.
 * @return	{function}	A debounced wrapper function around the specified function.
 */
export default function(fn, interval) {
	const time_last = 0;
	return (...args) => {
		const now = new Date();
		if(now - time_last > interval)
			fn(...args);
	}
}