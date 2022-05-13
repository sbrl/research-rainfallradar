"use strict";

import path from 'path';
import fs from 'fs';

import CliParser from 'applause-cli';

import log from './lib/io/NamespacedLog.mjs'; const l = log("core");
import { LOG_LEVELS } from './lib/io/Log.mjs';
import a from './lib/io/Ansi.mjs';

import settings from './settings.mjs';

const __dirname = import.meta.url.slice(7, import.meta.url.lastIndexOf("/"));

async function load_subcommands(cli) {
	let dirs = await fs.promises.readdir(path.join(__dirname, "subcommands"));
	for(let dir of dirs) {
		(await import(path.join(__dirname, "subcommands", dir, `meta.mjs`))).default(cli);
	}
}

export default async function () {
	let cli = new CliParser(path.resolve(__dirname, "../package.json"));
	cli.argument("verbose", "Enable verbose debugging output", null, "boolean")
		.argument("log-level", "Sets the log level. Value values: DEBUG, INFO (the default), LOG, WARN, ERROR, NONE", "INFO", "string");
	
	await load_subcommands(cli);
	
	let cli_parsed = cli.parse(process.argv.slice(2));
	
	if(cli.current_subcommand == null)
		cli.write_help_exit();
	
	Object.assign(settings, cli_parsed);
	
	l.level = LOG_LEVELS[settings.log_level];
	
	let subcommand_file = path.join(
		__dirname,
		"subcommands",
		cli.current_subcommand,
		`${cli.current_subcommand}.mjs`
	);
	
	if(!fs.existsSync(subcommand_file)) {
		l.error(`Error: The subcommand '${cli.current_subcommand}' doesn't exist (try --help for a list of subcommands and their uses).`);
		process.exit(1);
	}
	
	try {
		await (await import(subcommand_file)).default();
	}
	catch(error) {
		console.error();
		if(settings.cli.verbose)
			throw error;
		else
			console.error(`${a.fred}${a.hicol}${error.message}${a.reset}`);
		process.exit(1);
		throw error;
	}
}
