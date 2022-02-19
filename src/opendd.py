import pathlib
import sqlite3
import logging
from typing import Optional

import pandas as pd
from tqdm import tqdm
from joblib import Memory


# cache results
memory = Memory(".cache", verbose=0)

# vehicle classes from `CLASS` column
VEHICLE_CLASSES = ["Bus", "Car", "Medium Vehicle", "Heavy Vehicle", "Motorcycle"]
OTHER_CLASSES = ["Pedestrian", "Trailer"]


@memory.cache
def extract_samples_from_sqlite(p: pathlib.Path, debug: bool = False, 
    logger: Optional[logging.Logger] = None) -> pd.DataFrame:

    con = sqlite3.connect(str(p))
    cursor = con.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [r[0] for r in cursor.fetchall()]

    if debug:
        tables = tables[:2]

    if logger:
        logger.info(f"Working with {len(tables)} measurements.")

    df_list = list()
    for t in tqdm(tables, total=len(tables)):
        df_table = pd.read_sql(f'SELECT * FROM {t:s}', con)
        df_table["MEASUREMENT"] = t
        df_list.append(df_table)

    df = pd.concat(df_list)

    del df["TRAILER_ID"] 

    df["CLASS"] = df["CLASS"].astype("category")
    df["MEASUREMENT"] = df["MEASUREMENT"].astype("category")

    # keep only motorized vehicles
    df = df[df["CLASS"].isin(VEHICLE_CLASSES)]

    con.close()
    return df
