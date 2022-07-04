"use strict";

export default function(cli) {
	cli.subcommand("uniq", "Deduplicates the entries in a directory of recordified files.")
		.argument("source", "Path to the source directory.", null, "string")
		.argument("target", "Path to the target directory.", null, "string")
		.argument("count-file", "The number of records to store in each record file. See the documentation for the optimal value of this number (default: 4096).", 64*64);
}
