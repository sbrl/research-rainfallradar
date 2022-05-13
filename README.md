# Rainfall Radar

> A model to predict water depth data from rainfall radar information.

This is the 3rd major version of this model.





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
