"use strict";

export default function(cli) {
	cli.subcommand("recordify", "Converts rainfall radar and water depth data to a directory of .jsonl.gz files.")
		.argument("water", "Path to the water depths file, formatted as a stream of terrain50 objects. May or may not be gzipped.", null, "string")
		.argument("rainfall", "Path to the rainfall radar data, formatted as jsonl. May or may not be gzipped.",  null, "string")
		.argument("count-file", "The number of records to store in each record file. See the documentation for the optimal value of this number (default: 4096).", 64*64, "integer")
		.argument("rainfall-pattern", "The pattern of the number of time steps to average, as a comma-separated list of numbers. Given a point in time, each successive number specified works BACKWARDS from that point. For example, 1,4,10 would be 3 channels: 1 time step on it's own, then average the next 4 time steps, then average the next 10 steps (default: 1,3,3,5,12,24,48).", [1,3,3,5,12,24,48].reverse(), function(value) {
			return value.split(",")
				.map(el => parseInt(el))
				.reverse();
		})
		.argument("water-offset", "Make the water depth data be this many time steps ahead of the rainfall radar data. (default: 1)", 1, "integer")
		.argument("threshold", "Binarise the data according to the given threshold. Defaults to null, which means no thresholding is performed.", null, "float")
		.argument("output", "The path to the directory to write the generated .jsonl.gz files to.", null, "string");
}
