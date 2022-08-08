"use strict";

export default function(cli) {
	cli.subcommand("jsonl2tfrecord", "Convert a directory of .jsonl.gz files to .tfrecord.gz files.")
		.argument("source", "Path to the source directory.", null, "string")
		.argument("target", "Path to the target directory.", null, "string");
}
