# rainfallwrangler

> Wrangles rainfall radar and water depth data into something sensible.

This Node.js-based tool is designed for wrangling rainfall, heightmap, and water depth data into something that the image semantic segmentation model that is the main feature of this repository can understand.

The reason for this is efficiency: nothing less than a set of `.tfrecord` files for reading in parallel is sufficient if one wants the model to train in a reasonable length of time.


## System requirements
 - Linux (Windows *may* work but is untested. You will probably have a bad day if you use Windows)
 - [Node.js](https://nodejs.org/en) v16+
 - Python 3.8+ (encoding .tfrecord files, as all existing `npm` packages fo  doing this *suck*)
 - Experience with the terminal
 - Lots of time and patience


## Getting started
This tool, unlike [`nimrod-data-downloader`](https://www.npmjs.com/package/nimrod-data-downloader) and [`terrain50-cli`](https://www.npmjs.com/package/terrain50-cli), is not published to `npm`. This is because of the rather niche use-case this tool has.

To get started, first clone this git repository:

```bash
git clone git@github.com:sbrl/research-rainfallradar.git;
cd research-rainfallradar/rainfallwrangler;
```

Then, install dependencies:

```bash
npm install
pip3 install --user -r requirements.txt
```

The entrypoint for the tool is at `src/index.mjs`. Call it like so:

```bash
src/index.mjs --help
```

It has 4 subcommands:

- **recordify:** Converts a `.asc` heightmap, a concatenated `.asc` water depths file (output from [HAIL-CAESAR](https://github.com/sbrl/HAIL-CAESAR)), and a [`nimrod-data-downloader`](https://www.npmjs.com/package/nimrod-data-downloader) rainfall radar directory into an intermediate `.jsonl.gz` dataset. Defaults to putting 4096 samples per file.
- **uniq:** Deduplicates samples across an entire `.jsonl.gz` dataset. Basically hashes all samples with SHA256, marks duplicate hashes for deletion, and then files through all files in the dataset to remove those slated for deletion.
- **recompress:** Recompresses a `.jsonl.gz` dataset to ensure that (by default, 4096) samples are in each file. Needed after `uniq` since `uniq` can leave different numbers of records in each file.
- **jsonl2tfrecord:** Converts the aforementioned `.jsonl.gz` dataset into a `.tfrecord` dataset that the DeepLabV3+ model can understand

All of these subcommands, where possible, operate in parallel. The general workflow is:

1. `recordify`
2. `uniq`
3. `recompress`
4. `jsonl2tfrecord`

Full help for each command is available if you call `--help`:

```bash
src/index.mjs --help # Show general help for everything
src/index.mjs recordify --help # Snow specific help for the recordify subcommand
```


## Contributing
Contributions are very welcome - both issues and pull requests! Please mention in any pull requests that you release your work under the AGPL-3 (see below).


## Licence
Same as that of the main repository. All the code in this repository is released under the GNU Affero General Public License 3.0 unless otherwise specified. The full license text is included in the [`LICENSE.md` file](./LICENSE.md) in this repository. GNU [have a great summary of the licence](https://www.gnu.org/licenses/#AGPL) which I strongly recommend reading before using this software.