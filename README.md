# Rainfall Radar

> A model to predict water depth data from rainfall radar information.

This is the 3rd major version of this model.

Unfortunately using this model is rather complicated and involves a large number of steps. This README (will) explain it the best I can though.


## System Requirements
 - Linux (Windows *may* work but is untested. You will probably have a bad day if you use Windows)
 - Node.js (a *recent* version - i.e. v16+ - the version in the default Ubuntu repositories is too old)
 - Python 3.8+
 - Nvidia GPU (16GiB RAM+ is **strongly recommended**) + CUDA and CuDNN (see [this table](https://www.tensorflow.org/install/source#gpu) for which versions you need)
 - Experience with the command line
 - 1TiB disk space free
 - Lots of time and patience

## Overview
The process of using this model is as follows.

1. Apply for access to [CEDA's 1km rainfall radar dataset](https://catalogue.ceda.ac.uk/uuid/27dd6ffba67f667a18c62de5c3456350)
2. Obtain rainfall radar data (use [`nimrod-data-downloader`](https://www.npmjs.com/package/nimrod-data-downloader))
3. Obtain a heightmap (or *Digital Elevation Model*, as it's sometimes known) from the Ordnance Survey (can't remember the link, please PR to add this)
4. Use [`terrain50-cli`](https://www.npmjs.com/package/terrain50-cli) to slice the the output from steps #2 and #3 to be exactly the same size [TODO: Preprocess to extract just a single river basin from the data]
5. Push through [HAIL-CAESAR](*https://github.com/sbrl/HAIL-CAESAR) (this fork has the ability to handle streams of .asc files rather than each time step having it's own filename)
6. Use `rainfallwrangler` in this repository (finally!) to convert the output to .tfrecord files
7. Pretrain a contrastive learning model
8. Encode the rainfall radar data with the contrastive learning model you pretrained
9. Train the *actual* model to predict water depth

Only steps #6 to #9 actually use code in this repository.

## rainfallwrangler
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