"""Utilities used in the driving-off analysis.
"""

from dataclasses import dataclass
import logging
from typing import Optional, List
import multiprocessing as mp
from joblib import Parallel

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString

from .path_extraction import extract_drivable_paths
from .opendd import VEHICLE_CLASSES
from .preprocess import VehicleState, identify_driving_state, transform_df_to_trajectory_gdf
from .utils import pairwise, create_logger
from .trajectory import Trajectory
from .path_assignment import DrivablePath, find_nearest_path
from .reference_path import DiscreteReferencePath

logger = create_logger(__name__)

@dataclass
class DriveOffSituation:
    """Analysis artifact representing a drive-off sitation
    """
    
    # object IDs
    o1_id: int
    o2_id: int
    
    t: float
    
    # distance between the objects
    distance: float
    
    o2_state: VehicleState
    o2_timedelta_drive_off: Optional[float] 
    o2_velocity: float


def identify_driveoff_times(states: pd.Series) -> List[float]:
    """Identify points in time where a vehicle drives off.

    Args:
        states (pd.Series): object's states

    Returns:
        List[float]: points in time where an object changes from STANDING to DRIVING.
    """
    
    state_change = np.diff(states)
    
    d = VehicleState.DRIVING.value - VehicleState.STANDING.value
    state_change_indices = np.nonzero(state_change == d)[0] + 1
    
    if len(state_change_indices) < 1:
        return []
    else:
        res_list = list()
        for i in state_change_indices:
            res_list.append(states.index[i])
        return res_list


def find_drive_offs(traj1: Trajectory, traj2: Trajectory) -> List[DriveOffSituation]:
    """Analyze behaviour of vehicle 2 in situations where the vehicle 1 drives off.

    Args:
        traj1 (Trajectory): trajectory of first vehicle
        traj2 (Trajectory): trajectory of second vehicle

    Raises:
        ValueError: if states provided in the trajectory are not valid

    Returns:
        List[DriveOffSituation]: list with situations
    """
    
    common_time_range =  traj1.temporal_intersection(traj2) 
    if common_time_range[0] > common_time_range[1]:
        return list()
    
    o1_veh_states = traj1.get("STATE")
    o2_veh_states = traj2.get("STATE")

    # vehicle lengths
    o1_length = traj1.get("LENGTH").iloc[0]
    o2_length = traj2.get("LENGTH").iloc[0]
    
    # get velocity
    o2_velocities = traj2.get("V")
    
    o1_drive_off_times = identify_driveoff_times(o1_veh_states)
    o2_drive_off_times = identify_driveoff_times(o2_veh_states)
    
    if len(o1_drive_off_times) < 1:
        # return empty list
        return list()
    
    o2_drive_off_times = np.array(o2_drive_off_times)
    
    situations = list()
    
    # for each drive-off of the first object
    for t_drive_off in o1_drive_off_times:
        
        o1_state = traj1.sample(t_drive_off)
        o1_position_lon = o1_state["S"]
        
        # sample state of the second
        o2_state = traj2.sample(t_drive_off)
        
        o2_driving_state = int(o2_state["STATE"])
        
        o2_position_lon = o2_state["S"]
        o2_velocity = o2_state["V"]
        
        # consder the vehicle length
        distance = (o1_position_lon - o1_length) - (o2_position_lon + o2_length)
        
        if o2_driving_state == VehicleState.STANDING:
            try:
                time_drive_off_after_first = o2_drive_off_times[o2_drive_off_times >= t_drive_off][0]
            except IndexError:
                # probably the second vehicle never moves
                continue
            
            time_delta = time_drive_off_after_first - t_drive_off
            
            situation = DriveOffSituation(traj1.name, traj2.name, t_drive_off, distance, 
                                            VehicleState.STANDING, time_delta, o2_velocity)
            
            situations.append(situation)
        
        elif o2_driving_state == VehicleState.DRIVING:
            
            o2_velocities
            situation = DriveOffSituation(traj1.name, traj2.name, t_drive_off, distance, 
                                            VehicleState.DRIVING, None, o2_velocity)
            situations.append(situation)
        else:
            raise ValueError("Interesting ...")
    return situations


def analyze_driveoffs_from_path(path_trajectories: gpd.GeoDataFrame) -> pd.DataFrame:
    # Assumptions: NO OVERTAKES!
    
    # identify the starting (long.) location of each trajectory for sorting
    # would fail in the first second, if not done
    path_trajectories["START_LON"] = path_trajectories["frenet_path"].apply(lambda ls: ls.xy[0][0])
    
    path_trajectories = path_trajectories.sort_values(["START_TIME", "START_LON"], ascending=[True, False])
    
    situations = list()
    for i, (o1, o2) in enumerate(pairwise(range(len(path_trajectories)))):
        o1_row = path_trajectories.iloc[o1]
        o2_row = path_trajectories.iloc[o2]
        
        cols = ["S", "V", "ACC", "STATE", "LENGTH"]
        traj1, traj2 = Trajectory.from_trace(o1_row, cols), Trajectory.from_trace(o2_row, cols)
        
        drive_off_situations = find_drive_offs(traj1, traj2)
        situations += drive_off_situations
        
    df_situations = pd.DataFrame(situations)
    df_situations["path_id"] = path_trajectories["path_id"].iloc[0]
        
    return df_situations


def _trace_to_frenet(trace: pd.Series, reference_paths: List[DiscreteReferencePath]) -> LineString:
    path_id = trace["path_id"]
    ref_path = reference_paths[path_id]
    return ref_path.linestring_to_frenet(trace["geometry"])


def extract_moving_off_situations(roundabout_samples: pd.DataFrame, trafficlanes: gpd.GeoDataFrame,
                                num_cores: int = 1, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """Extract moving off situations from raw data table provided in OpenDD dataset.

    Args:
        roundabout_samples (pd.DataFrame): raw, unprocessed data.
            Refer to https://l3pilot.eu/data/opendd.
        trafficlanes (gpd.GeoDataFrame): traffic lanes geometric description (linestrings)
        logger (Optional[logging.Logger], optional): Logger for the output. Defaults to None.

    Returns:
        pd.DataFrame: moving-off situations as records
    """

    paths = extract_drivable_paths(trafficlanes)

    if logger:
        logger.info(f"{len(paths)} reference paths could be extracted!")    

    if num_cores > 1:

        if logger:
            logger.info(f"{num_cores} CPUs cores will be used.")

        with mp.Pool(num_cores) as pool:

            async_result_list = list()
            for meas, df_measurement in roundabout_samples.groupby("MEASUREMENT"):
                res = pool.apply_async(_extract_moving_off_situations_single_measurement, \
                    args=[df_measurement, paths, meas, logger])
                async_result_list.append(res)
            
            result_list = [res.get(timeout=None) for res in async_result_list]
        
    else:
        # old implementation
        result_list = list()
        for meas, df_measurement in roundabout_samples.groupby("MEASUREMENT"):
            df_situations = _extract_moving_off_situations_single_measurement(df_measurement, paths, meas, logger)
            result_list.append(df_situations)
            
    df_situations = pd.concat(result_list)
    df_situations["measurement"] = df_situations["measurement"].astype("category")
    
    return df_situations


def _extract_moving_off_situations_single_measurement(df_measurement: pd.DataFrame, paths: List[DrivablePath], 
    measurement_name: str, logger = None):

    if len(df_measurement) < 1:
        return pd.DataFrame()
    
    gdf_traces = transform_df_to_trajectory_gdf(df_measurement)

        # keep only motorized vehicles
    gdf_traces = gdf_traces[gdf_traces["CLASS"].isin(VEHICLE_CLASSES)]

        # identify driving state
    gdf_traces["STATE"] = gdf_traces.apply(lambda row: identify_driving_state(row), axis=1)
        
    if logger:
        logger.info(f"{measurement_name}: assigning {len(gdf_traces):d} trajectories to path")

        # assign to reference path
    gdf_traces["path_id"] = gdf_traces.apply(lambda row: find_nearest_path(row["geometry"], paths, N=25), axis=1)

        # create discrete reference paths
    reference_paths = [DiscreteReferencePath.from_linestring(dp.as_linestring(), resolution=0.05) \
                           for dp in paths]

        # transform to frenet (frenet_path is a linestring, x=tangential, y=normal)
    gdf_traces["frenet_path"] = gpd.GeoSeries(gdf_traces.apply(lambda row: _trace_to_frenet(row, reference_paths), axis=1))

        # extract the first frenet coordinate
    gdf_traces["S"] = gdf_traces["frenet_path"].apply(lambda row: row.xy[0])

        # for each path, find the drive-off situations
    if logger:
        logger.info(f"{measurement_name}: Retrieving drive-off situations.")
    
    df_situations = gdf_traces.groupby("path_id").apply(analyze_driveoffs_from_path)
    df_situations = df_situations.reset_index(drop=True)

    if len(df_situations) < 1:
        return pd.DataFrame()

    df_situations["o1_id"] = df_situations["o1_id"].astype(int)
    df_situations["o2_id"] = df_situations["o2_id"].astype(int)
    df_situations["o2_state"] = df_situations["o2_state"].astype(int)
        
        # record measurement name
    df_situations["measurement"] = measurement_name

    print(f"'{measurement_name}' done!")
    return df_situations
