"""Command line tool for extracting driving off situations from a single roundabout.

See OpenDD dataset for details.

"""
import sys
import pathlib
import warnings

import click
import pandas as pd
import geopandas as gpd
from tqdm.auto import tqdm
from shapely.errors import ShapelyDeprecationWarning

sys.path.append(".")
from src.path_extraction import traffic_lanes_to_graph, \
    extract_paths_from_graph, driving_path_overview_plot
from src.opendd import extract_samples_from_sqlite
from src.utils import create_logger
from src.drive_off import extract_moving_off_situations

tqdm.pandas()
logger = create_logger(__name__)

# unfortunately we have to ignore it
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

@click.command()
@click.argument("roundabout_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path(exists=False))
@click.option('--debug', is_flag=True)
def main(roundabout_path: str, output_path: str, debug: bool):

    # easier handling with pathlib
    roundabout_path = pathlib.Path(roundabout_path)
    output_path = pathlib.Path(output_path)

    if not output_path.is_dir():
        output_path.mkdir()

    roundabout_name = roundabout_path.name
    logger.info(f"Extracting reference paths from '{roundabout_name}'")
    
    shapefiles_trafficlanes_path = roundabout_path / f"map_{roundabout_name}/shapefiles_trafficlanes"

    trafficlanes = gpd.read_file(shapefiles_trafficlanes_path)
    G = traffic_lanes_to_graph(trafficlanes)
    
    paths = extract_paths_from_graph(G)
    logger.info(f"{len(paths)} reference paths could be extracted!")

    fig = driving_path_overview_plot(trafficlanes, paths)
    fig.suptitle(roundabout_name, y=1.01, fontsize=15)

    roundabout_paths_overview_plot_path = output_path / f"{roundabout_name}_reference_paths.png"
    fig.savefig(roundabout_paths_overview_plot_path, dpi=200)

    # Extract database
    logger.info(f"Extracting data from sqlite database.")
    roundabout_db = list(roundabout_path.glob("*.sqlite"))[0]    
    df_roundabout = extract_samples_from_sqlite(roundabout_db, debug)

    if logger:
        num_measurements = df_roundabout["MEASUREMENT"].nunique()
        logger.info(f"Found {num_measurements:d} measurements.")

    df_situations = extract_moving_off_situations(df_roundabout, paths, logger)
    df_situations["roundabout"] = roundabout_name

    situations_output_path = output_path / "situations.csv"
    df_situations.to_csv(situations_output_path, index=False)


if __name__ == '__main__':
    main()