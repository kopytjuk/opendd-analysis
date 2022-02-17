"""Utilities used in the driving-off analysis.
"""

from dataclasses import dataclass
from typing import Optional, List
from itertools import tee

import pandas as pd
import numpy as np
import geopandas as gpd

from .preprocess import VehicleState
from .utils import pairwise, create_logger
from .trajectory import Trajectory

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
        
        distance = o1_position_lon - o2_position_lon 
        
        if o2_driving_state == VehicleState.STANDING:
            time_drive_off_after_first = o2_drive_off_times[o2_drive_off_times >= t_drive_off][0]
            
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
        
        cols = ["S", "V", "ACC", "STATE"]
        traj1, traj2 = Trajectory.from_trace(o1_row, cols), Trajectory.from_trace(o2_row, cols)
        
        drive_off_situations = find_drive_offs(traj1, traj2)
        situations += drive_off_situations
        
    df_situations = pd.DataFrame(situations)
        
    return df_situations