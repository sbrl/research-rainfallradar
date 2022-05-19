"use strict";

/**
 * Transpose a 2D array. In other words, rotate the contents of the array 90°
 * *anti*clockwise.
 * Note that a normal transpose would rotate 90° clockwise, not *anti*clockwise.
 * Adapted from https://stackoverflow.com/a/17428705/1460422 to rotate it anti-clockwise, not clockwise
 * Note that even though `.reverse()` is a mutating function, it's ok here because `.map()` creates a new array.
 * This was written because in the NimrodRunner we made the embarrasing mistake
 * of winding up writing the data the wrong way around.....
 * Imported from nimrod-data-downloader.
 * @param	{any[][]}	data	The 2D array to transpose.
 * @return	{any[][]}	The transposed array.
 */
function transpose(data) {
	return data[0].map((_col, i) => data.map(row => row[i])).reverse();
}

export default transpose;
