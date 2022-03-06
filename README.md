This repo is the foundation for my blog article: [LINK](https://kopytjuk.github.io/posts/open-dd-analysis/)

# OpenDD dataset analysis

![logo](logo.png)

<p><small>© Image Copyright 2021. L3Pilot Consortium</small></p>

Explore the *Large-Scale Roundabout Drone Dataset* ([OpenDD](https://l3pilot.eu/data/opendd)) regarding interesting traffic situations.

Paper: https://arxiv.org/pdf/2007.08463.pdf

## Vehicle moving off analysis

### Notebooks

- Prototyping notebook with helpful visualization. Step-by-step guide for what happens in the CLI tool below: [Distance_Analysis_Prototyping.ipynb](notebooks/Distance_Analysis_Prototyping.ipynb)
- Results notebook (TBD)

### CLI tool

Run to extract all driving-off situations from a singe roundabout

```
python tools/extract_moving_off.py data/raw/rdb1 test-output --debug
```