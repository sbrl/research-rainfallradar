# Rainfall Radar

> A model to predict water depth data from rainfall radar information.

This is the 3rd major version of this model.

Unfortunately using this model is rather complicated and involves a large number of steps. There is no way around this. This README (will) explain it the best I can though.

> [!WARNING]
> This README is currently under construction!

## Paper
The research in this repository has been published in a conference paper(!)

- **Title:** Towards AI for approximating hydrodynamic simulations as a 2D segmentation task
- **Conference:** Northern Lights Deep Learning Conference 2024
- **DOI:** coming soon, but in advance you can view what should be the final paper here: <https://openreview.net/pdf?id=TpOsdB4gwR>

**Abstract:**  
Traditional predictive simulations and remote sensing techniques for forecasting floods are based on fixed and spatially restricted physics-based models. These models are computationally expensive and can take many hours to run, resulting in predictions made based on outdated data. They are also spatially fixed, and unable to scale to unknown areas.

By modelling the task as an image segmentation problem, an alternative approach using artificial intelligence to approximate the parameters of a physics-based model in 2D is demonstrated, enabling rapid predictions to be made in real-time.


## System Requirements
 - Linux (Windows *may* work but is untested. You will probably have a bad day if you use Windows)
 - Node.js (a *recent* version - i.e. v16+ - the version in the default Ubuntu repositories is too old)
 - Python 3.8+
 - Nvidia GPU (16GiB RAM+ is **strongly recommended**) + CUDA and CuDNN (see [this table](https://www.tensorflow.org/install/source#gpu) for which versions you need)
 - Experience with the command line
 - 1TiB disk space free
 - Lots of time and patience

## Overview
The process of using this model is as as illustrated:

![Flowchart illustrating the data flow for using the code in this repository to make predictions water depth](./research-rainfallradar%20overview.png)

More fully:

1. Apply for access to [CEDA's 1km rainfall radar dataset](https://catalogue.ceda.ac.uk/uuid/27dd6ffba67f667a18c62de5c3456350)
2. Download 1km rainfall radar data (use [`nimrod-data-downloader`](https://www.npmjs.com/package/nimrod-data-downloader))
3. Obtain a heightmap (or *Digital Elevation Model*, as it's sometimes known) from the Ordnance Survey (can't remember the link, please PR to add this)
4. Use [`terrain50-cli`](https://www.npmjs.com/package/terrain50-cli) to slice the the output from steps #2 and #3 to be exactly the same size [TODO: Preprocess to extract just a single river basin from the data]
5. Push through [HAIL-CAESAR](https://github.com/sbrl/HAIL-CAESAR) (this fork has the ability to handle streams of .asc files rather than each time step having it's own filename)
6. Use `rainfallwrangler` in this repository (finally!) to convert the output to .json.gz then .tfrecord files
7. Train a DeepLabV3+ prediction model

Only steps #6 and #7 actually use code in this repository. Steps #2 and #4 involve the use of modular [`npm`](https://npmjs.org/) packages.

### Obtaining the data
The data in question is the Met Office's NIMROD 1km rainfall radar dataset, stored in the CEDA archive. It is updated every 24 hours, and has 1 time step every 5 minutes.

The data can be found here: <https://catalogue.ceda.ac.uk/uuid/27dd6ffba67f667a18c62de5c3456350>

There is an application process to obtain the data. Once complete, use the tool `nimrod-data-downloader` to automatically download & parse the data:

<https://www.npmjs.com/package/nimrod-data-downloader>

This tool was also written me, [@sbrl](https://starbeamrainbowlabs.com/) - the primary author on the paper mentioned above.

Full documentation on this tool is available at the above link.

**Heightmap:** Anything will do, but I used the [Ordnance Survey Terrain50](https://www.ordnancesurvey.co.uk/products/os-terrain-50) heightmap, since it is in the OS National Grid format (eww >_<), same as the aforementioned rainfall radar data.

### Running the simulation
Once you have your data, ensure it is in a format that the HAIL-CAESAR model will understand. For the rainfall radar data, this is done using the `radar2caesar` command of `nimrod-data-downloader`, as mentioned above.

before running the simulation, the heightmap and rainfall radar will need cropping to match one another. For this the tool [`terrain50-cli`](https://www.npmjs.com/package/terrain50-cli) was developed.

Once this is done, the next step is to run HAIL-CAESAR. Details on this can be found here:

<https://github.com/sbrl/HAIL-CAESAR/>

....unfortunately, due to the way HAIL-CAESAR is programmed, it reads *all* the rainfall radar data into memory first before running the simulation. From memory for data from 2006 to 2020 it used approximately 350GiB - 450GiB RAM.

Replacing this simulation with a better one is on the agenda for moving forwards with this research project - especially since I need to re-run a hydrological simulation anyway when attempting a tile-based approach.

### Preparing to train the model
Once the simulation has run to completion, all 3 pieces are now in place to prepare to train an AI model. The AI model training process requires that data is stored in `.tfrecord` files for efficiency given the very large size of the dataset in question.

This is done using the `rainfallwrangler` tool in the eponymous directory in this repository. Full documentation on `rainfallwrangler` can be found in the README in that directory:

[rainfallwrangler README](./rainfallwrangler/README.md)

`rainfallwrangler` is a Node.js application to wrangle the dataset into something more appropriate for training an AI efficiently. The rainfall radar and water depth data are considered temporally to be regular time steps. Here's a diagram explaining the terminology:

```
                       NOW
│                       │         │Water depth
│▼ Rainfall Radar Data ▼│[Offset] │▼
├─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┼─┬─┬─┬─┬─┼─┐
│ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │
│ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │
│ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │
│ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │
└─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┼─┴─┴─┴─┴─┴─┘
                        │
◄────────── Timesteps ─────────────►
```

Note to self: 150.12 hashes/sec on i7-4770 4c8t, ???.?? hashes/sec on Viper compute

After double checking, rainfallwrangler does NOT mess with the ordering of the data.



### Training the model
After all of the above steps are completed, a model can now be trained.

The current state of the art (that was presented in the above paper!) is based on DeepLabV3+. A note of caution: this repository contains some older models, so it can be easy to mix them up. Hence this documentation :-)

<------ WRITING HERE

TODO: Continue the guide here.


## License
All the code in this repository is released under the GNU Affero General Public License unless otherwise specified. The full license text is included in the [`LICENSE.md` file](./LICENSE.md) in this repository. GNU [have a great summary of the licence](https://www.gnu.org/licenses/#AGPL) which I strongly recommend reading before using this software.
