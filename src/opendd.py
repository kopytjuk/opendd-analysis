import pathlib
import sqlite3

import pandas as pd
from tqdm import tqdm


# vehicle classes from `CLASS` column
VEHICLE_CLASSES = ["Bus", "Car", "Medium Vehicle", "Heavy Vehicle", "Motorcycle"]
OTHER_CLASSES = ["Pedestrian", "Trailer"]


def extract_samples_from_sqlite(p: pathlib.Path) -> pd.DataFrame:

    con = sqlite3.connect(str(p))
    cursor = con.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [r[0] for r in cursor.fetchall()]

    print(f"Found {len(tables)} measurements")

    df_list = list()
    for t in tqdm(tables, total=len(tables)):
        df_table = pd.read_sql(f'SELECT * FROM {t:s}', con)
        df_table["table"] = t
        df_list.append(df_table)

    df = pd.concat(df_list)

    del df["TRAILER_ID"] 

    df["CLASS"] = df["CLASS"].astype("category")

    # keep only motorized vehicles
    df = df[df["CLASS"].isin(VEHICLE_CLASSES)]

    con.close()
    return df
