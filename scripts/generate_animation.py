import sys
import pathlib

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from tqdm import tqdm

sys.path.append(".")
from src.preprocess import preprocess
from src.utils import WorldDefinition
from src.visualize import visualize_timestep

FREQUENCY = 30

data_path = "data/raw/rdb1.parquet"
df = pd.read_parquet(data_path)

print("Raw data loaded!")

df = df[df["table"] == "rdb1_1"]

gdf = preprocess(df, FREQUENCY)
gdf = gdf.set_geometry("bbox")

print("Preprocessing successful!")

# get traffic lanes
trafficlanes_shapefile = pathlib.Path("data/raw/rdb1/map_rdb1/shapefiles_trafficlanes")
trafficlanes = gpd.read_file(trafficlanes_shapefile)

print("Trafficlanes loaded!")

world_def = WorldDefinition.from_pgw_file(pathlib.Path(r'data\raw\rdb1\geo-referenced_images_rdb1\rdb1.pgw'))
extent = world_def.get_image_extent(3840, 2160)

print("World file loaded!")

fig, ax = plt.subplots(figsize=(16, 9), facecolor="w")

moviewriter = FFMpegWriter(FREQUENCY)
moviewriter.setup(fig, 'test.mp4', dpi=200)

gdf = gdf.query("k < 300")

for k, timestep_data in tqdm(gdf.groupby("k"), total=gdf["k"].nunique()):

    ax.clear()

    visualize_timestep(ax, timestep_data, trafficlanes)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    moviewriter.grab_frame()

moviewriter.finish()
