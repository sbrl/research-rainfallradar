"use strict";

import fs from 'fs';
import path from 'path';
import os from 'os';

import workerpool from 'workerpool';
import p_map from 'p-map';
import p_reflect from 'p-reflect';
import pretty_ms from 'pretty-ms';

import log from '../io/NamespacedLog.mjs'; const l = log("recorduniq:manager");
import records_recompress from './records_recompress.mjs';


const __dirname = import.meta.url.slice(7, import.meta.url.lastIndexOf("/"));

class RecordUniqManager {
	constructor(items_per_file) {
		this.items_per_file = items_per_file;
		
		this.worker_count = os.cpus().length;
		this.pool = workerpool.pool(path.join(__dirname, "record_uniq_worker/worker.mjs"), {
			maxQueueSize: 100,
			maxWorkers: this.worker_count
		});
		
		this.hashes = new Map();
		this.init_complete = false;
	}
	
	async init() {
		if(this.init_complete) return;
		
		this.proxy = await this.pool.proxy();
		
		this.init_complete = true;
	}
	
	async deduplicate(dirpath_source, dirpath_target) {
		await this.init();
		const time_start = new Date();
		this.hashes.clear();
		this.items_deleted = 0;
		
		const files = (await fs.promises.readdir(dirpath_source))
			.filter(filename => filename.endsWith(".jsonl.gz"))
			.map(filename => path.join(dirpath_source, filename));
		
		l.log(`STEP [1 / 5]: Hashing files`);
		await p_map(files, this.#do_single_hash.bind(this), { concurrency: this.worker_count + 10 });
		l.log(`STEP [1 / 5]: ${this.hashes.size} hashes gathered in total.`);
		
		l.log(`STEP [ 2 / 5 ]: Identify duplicates`);
		const dupes = this.find_duplicates();
		this.hashes.clear(); // Save memory
		l.log(`STEP [ 2 / 5 ]: ${dupes.length} duplicate groups identified`);
		
		l.log(`STEP [ 3 / 5 ]: Assemble deletion lists`);
		const deletion_lists = this.assemble_deletion_lists(dupes);
		console.log(deletion_lists);
		l.log(`STEP [ 3 / 5 ]: ${[...deletion_lists.values()].reduce((acc, next) => next.length + acc, 0)} duplicates to be deleted.`);
		
		l.log(`STEP [ 4 / 5 ]: Delete duplicates`);
		await p_map(
			deletion_lists.entries(),
			async (args) => await this.#do_single_delete(...args),
			{ concurrency: this.worker_count + 10 }
		);
		l.log(`STEP [ 4 / 5 ]: ${this.items_deleted} duplicates deleted.`);
		
		l.log(`STEP [ 5 / 5 ]: Recompress files`);
		const { recompress_lines, recompress_files } = await records_recompress(
			dirpath_source, dirpath_target ?? this.#adjacent_dir(dirpath_source),
			this.items_per_file
		);
		l.log(`STEP [ 5 / 5 ]: Complete with ${recompress_files} files ${recompress_lines} lines at final count.`);
		l.log(`Done in ${pretty_ms(new Date() - time_start)}, thank you :D`);
	}
	
	#adjacent_dir(dir, target="deduped") {
		const dirname = path.dirname(dir);
		const basename = path.basename(dir);
		return path.join(dirname, `${basename}-${tag}`);
	}
	
	find_duplicates() {
		const result = [];
		const hashes_seen = [];
		for(const [ id, hash ] of this.hashes.entries()) {
			if(hashes_seen.includes(hash)) continue;
			const dupes_group = [ { id, hash } ];
			for(const [ id_inner, hash_inner ] of this.hashes.entries()) {
				if(id === id_inner) continue;
				if(hash === hash_inner) dupes_group.push( { id: id_inner, hash: hash_inner });
			}
			hashes_seen.push(hash);
			if(dupes_group.length > 1) {
				result.push(dupes_group);
			}
		}
		
		return result;
	}
	
	assemble_deletion_lists(dupe_groups) {
		const result = new Map();
		for(const dupe_group of dupe_groups) {
			for(const dupe of dupe_group.slice(1)) { // Keep the first one
				const [ filename, i ] = dupe.id.split(`|`, 2);
				if(!result.has(filename)) result.set(filename, []);
				result.get(filename).push(parseInt(i, 10));
			}
		}
		return result;
	}
	
	async #do_single_hash(filepath) {
		if(filepath.includes("|")) throw new Error(`Filepath contains bar character: ${filepath}`);
		const filename = path.basename(filepath);
		
		// l.log(`Hashing ${path.basename(filepath)}`);
		const result = await p_reflect(this.proxy.hash_targets(filepath));
		if(result.isRejected) {
			l.warn(`Got error from worker when hashing ${filename}:`, result.reason);
			return;
		}
		
		for(const { i, hash } of result.value) {
			this.hashes.set(`${filepath}|${i}`, hash);
		}
	}
	
	async #do_single_delete(filename_source, deletion_list) {
		const result = await p_reflect(this.proxy.delete_duplicates(filename_source, deletion_list));
		if(result.isRejected) {
			l.warn(`Got error from worker when deleting ${deletion_list.length} entries from ${filename_source}:`, result.reason);
			return null;
		}
		
		this.items_deleted += result.value;
	}
}

export default RecordUniqManager;