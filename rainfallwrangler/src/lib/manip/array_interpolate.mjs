"use strict";

/**
 * Interpolates the values between 2 2D arrays.
 * Note that both arrays MUST be exactly the same size!
 * Imported from nimrod-data-downloader.
 * @param	{number[][]}	a			A 2D array of numbers.
 * @param	{number[][]}	b			A 2D array of numbers.
 * @param	{number}		percentage	The percentage of the way from a to b to use when interpolating
 * @return	{number[][]}	Another 2D array of numbers between the 2 different input arrays.
 */
function interpolate(a, b, percentage) {
	let result = [];
	for(let i = 0; i < a.length; i++) {
		let row = [];
		for(let j = 0; j < a[i].length; j++) {
			// If this row in b isn't defined, fill it in
			// Not sure why we'd have variable lengths here
			if(typeof b[i] == "undefined")
				b[i] = Array.apply(null, Array(a[i].length)).map(Number.prototype.valueOf, 0);
			/*
			 * Example:
			 * d = 10, e = 20, percentage = 0.75
			 * result = ((e - d) * percentage + d)
			 */
			row.push(((b[i][j] - a[i][j]) * percentage) + a[i][j]);
		}
		result.push(row);
	}
	return result;
}

export default interpolate;
