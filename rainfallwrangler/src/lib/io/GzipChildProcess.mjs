"use strict";

import EventEmitter from 'events';
import child_process from 'child_process';

import log from './NamespacedLog.mjs'; const l = log("gzipchildprocess");
import { end_safe } from './StreamHelpers.mjs';


/**
 * Spawns and manages a gzip child process.
 * @deprecated Use spawn-stream instead
 * @extends EventEmitter
 */
class GzipChildProcess extends EventEmitter {
	get stdin() { return this.child_process.stdin; }
	get stdout() { return this.child_process.stdout; }
	get stderr() { return this.child_process.stderr; }
	
	constructor(auto_start = true) {
		super();
		
		this.debug = false;
		this.child_process = null;
		
		this.has_ended = false;
		
		if(auto_start)
			this.start();
	}
	
	start() {
		if(this.child_process != null)
			throw new Error("Invalid Operation: Can't start the child process, since it's already been started.");
		
		this.child_process = child_process.spawn(
			"gzip", [], {
				// Pipe stdin + stdout; send error to the parent process
				stdio: [ "pipe", "pipe", "inherit" ]
			}
		);
		this.child_process.on("close", () => {
			if(this.debug) l.debug("Close event triggered");
			this.has_ended = true;
			this.emit("close");
		});
		// FUTURE: Perhaps just throwing the error would be a better choice?
		this.child_process.on("error", (error) => {
			this.emit("error", error);
		});
		
	}
	
	/**
	 * Returns a Promise that resolves when the gzip process gracefully exits.
	 * If the gzip child process has already exited, then it resolves immediately.
	 * Gracefully closes the process by closing stdin.
	 * @return	{Promise}
	 */
	async close() {
		if(this.debug) l.debug("end_gracefully called");
		if(this.has_ended) {
			if(this.debug) l.debug("It's been ended already - nothing to do");
			return;
		}
		if(!this.stdin.writableFinished) {
			if(this.debug) l.debug("Closing stdin");
			await end_safe(this.stdin);
			if(this.debug) l.debug("stdin closed successfully");
		}
		if(this.has_ended) {
			if(this.debug) l.debug("It's been ended already - nothing to do");
			return;
		}
		if(this.debug) l.debug("Waiting for close event");
		await EventEmitter.once(this, "close");
		if(this.debug) l.debug("Close event fired, our work is done");
	}
}

export default GzipChildProcess;
