# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 NautiPy Developers. All rights reserved.
Use of this source code is governed by the MIT license that can be found in
LICENSE.txt
"""

import numpy as np
from math import radians, cos, sin, asin, sqrt, degrees, atan2
from scipy.optimize import minimize



class Pos():
    """Class handling a single nautical (lat, lon) position
    """
    def __init__(self, lat: str or float, lon: str or float, desc:str=None):
        """
        Parameters
        ----------
        lat : str or float
            ISO 6709 latitude: ±DD.D  (e.g. +50.12257).
        lon : str or float
            ISO 6709 longitude: ±DDD.D (e.g. +008.66370).
        desc : str, optional
            Description or name of the position. The default is None.
        """
        self.lat = lat
        self.lon = lon
        self.convert_coordinates()
        self.desc = desc
        assert isinstance(self.lat, float) and isinstance(self.lon, float)
        assert -90 <= self.lat <= 90, 'Latitude out of range!'
        assert -180 <= self.lon <= 180, 'Longitude out of range!'
        
        
    def coordinates(self) -> tuple:
        """Return own (lat, lon)

        Might put some format conversion here

        Returns
        -------
        tuple
            Latitude, Longitude.
        """
        return (self.lat, self.lon)
        
        
    def convert_coordinates(self) -> None:
        """ToDO: Convert GPS input format to ISO 6709
        
        """
        #direction = {'N':1, 'S':-1, 'E': 1, 'W':-1}
        self.lat = self.lat 
        self.lon = self.lon
        
        
    def displace(self, heading:float, distance:float):
        """Displace position by distance in heading direction

        Parameters
        ----------
        heading : float
            Direction of displacement in degrees.
        distance : float
            Distance in kilometers.

        Returns
        -------
        Pos
            New position object.
        """
        theta = radians(heading)
        lat1 = radians(self.lat)
        lon1 = radians(self.lon)
        delta = distance / 6371
        lat_new = np.arcsin( np.sin(lat1) * np.cos(delta) +
                          np.cos(lat1) * np.sin(delta) * np.cos(theta) )
        lon_new = lon1 + np.arctan2( np.sin(theta) * np.sin(delta) * np.cos(lat1),
                                  np.cos(delta) - np.sin(lat1) * np.sin(lat_new))
        lon_new = (lon_new + 3 * np.pi) % (2 * np.pi) - np.pi
        return Pos(degrees(lat_new), degrees(lon_new))
            
        
def haversine(pos1:Pos, pos2:Pos) -> float:
    """Haversine distance
    
    The haversine formula determines the great-circle distance between two 
    points on a sphere given their longitudes and latitudes.

    Parameters
    ----------
    pos1 : Pos
        First position coordinates.
    pos2 : Pos
        Second position coordinates.

    Returns
    -------
    float
        Distance between both positions in kilometers.
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [pos1.lon, pos1.lat, 
                                           pos2.lon, pos2.lat])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    h = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    d = 2 * asin(sqrt(h)) 
    radius = 6371 # Radius of earth in kilometers
    return d * radius


def bearing(pos1:Pos, pos2:Pos, correction:float=0) -> float:
    """Absolute bearing brom pos1 to pos2.

    Parameters
    ----------
    pos1 : Pos
        First position coordinates.
    pos2 : Pos
        Second position coordinates.
    correction : float, optional
        Specify correction to return magnetic bearing. 
        The default is 0, i.e. true bearing.

    Returns
    -------
    float
        Bearing from pos1 to pos2.
    """
    lat1 = radians(pos1.lat)
    lat2 = radians(pos2.lat)
    diffLong = radians(pos2.lon - pos1.lon)
    x = sin(diffLong) * cos(lat2)
    y = cos(lat1) * sin(lat2) - (sin(lat1) * cos(lat2) * cos(diffLong))
    initial_bearing = atan2(x, y)
    # Now we have the initial bearing but math.atan2 return values
    # from -180° to + 180° which is not what we want for a compass bearing
    # The solution is to normalize the initial bearing as shown below
    initial_bearing = degrees(initial_bearing)
    bearing = (initial_bearing + 360) % 360
    return bearing + correction


def angle_between_bearings(bearing1:float, bearing2:float) -> float:
    """Return angle between two bearings
    
    Parameters
    ----------
    bearing1 : float
        Bearing in degrees.
    bearing2 : float
        Bearing in degrees.

    Returns
    -------
    angle
        Angle in degrees.
    """
    assert 0 <= bearing1 <= 360, 'bearing1 out of bounds!'
    assert 0 <= bearing2 <= 360, 'bearing2 out of bounds!'
    abs_diff = abs(bearing1 - bearing2)
    return min(abs_diff, abs(360 - abs_diff))


def opposite_bearing(bearing1:float) -> float:
    """Return the oppisite bearing, e.g. 90 -> 270
    
    Parameters
    ----------
    bearing1 : float
        Bearing in degrees.

    Returns
    -------
    bearing : float
        Opposite bearing in degrees.
    """
    return bearing1 - 180*(bearing1 > 180) + 180*(bearing1 <= 180)


def triangulate(station1:Pos, bearing1:float, station2:Pos, bearing2:float) -> Pos:
    """
    Trigonometric triangulation to find a position from its bearings 
    to two stations.
    
    Parameters
    ----------
    station1 : Pos
        First known position.
    bearing1 : float
        Bearing to first ficing.
    station2 : Pos
        Second known position.
    bearing2 : float
        Bearing to second known position.

    Returns
    -------
    pos3 : Pos
        Triangulated position.
    """
    # From point 1 to target
    bearing_13 = opposite_bearing(bearing1)
    # From point 2 to target
    bearing_23 = opposite_bearing(bearing2)
    # Angles
    angle_1 = angle_between_bearings(bearing_13, bearing(station1, station2))
    angle_2 = angle_between_bearings(bearing_23, bearing(station2, station1))
    angle_3 = 180 - angle_1 - angle_2
    # Distances via law of Sines: a / sin(A) = c / sin(C)
    distance_12 = haversine(station1, station2)
    distance_13 = distance_12 / sin(radians(angle_3)) * sin(radians(angle_2))
    #distance_23 = distance_12 / sin(radians(angle_3)) * sin(radians(angle_1))
    return station1.displace(bearing_13, distance_13)
    

def multilaterate(stations:list):
    """
    True range multilateration is a method to determine the location of a 
    movable vehicle or stationary point in space using multiple ranges 
    (stations) between the vehicle/point and multiple spatially-separated 
    known locations (often termed 'stations').

    Parameters
    ----------
    stations : list
        Position / distance pairs for N stations.
        stations = [(Pos(lat,lon), dist:float),
                     (Pos(lat,lon), dist:float),
                     (Pos(lat,lon), dist:float),
                     ...]

    Returns
    -------
    position
        Pos class object with multilaterated position.

    """
    assert len(stations) >= 3, 'I need >= 3 stations!'
    stations = np.array(stations)
    stations = stations[stations[:,1].argsort()]
    # We use the fixing with the shortest distance as initial guess
    x0 = stations[1,0].coordinates()
    # Simple OLS error function
    def error(x):
        current_pos =  Pos(x[0], x[1])
        error = 0
        for pos, dist in stations:
            error += (haversine(current_pos, pos) - dist)**2
        return error**0.5
    # Minimize squared errors
    position = minimize(error, x0, method='L-BFGS-B', 
                        options={'ftol':1e-5, 'maxiter': 1e+6})
    # Return the position object
    return Pos(position.x[0], position.x[1], desc=f'Error = {position.fun}')
    
    
    
#%% README.md
if __name__ == '__main__':
    ### Basics
    # Store and descibe your position
    work = Pos(50.127198, 8.665562, desc='Campus building')
    # Get relative position heading 90 degrees 12 kilometers away
    work_displaced = work.displace(90, 12)
    # Get bearing to position
    bearing(work, work_displaced)
    # Get distance to position
    haversine(work, work_displaced)
    
    ### Triangulation and Multilateration
    stations = {1:Pos(50.116135, 8.670277, 'Opernturm'),
                2:Pos(50.112836, 8.666753, 'Deka tower'),
                3:Pos(50.110347, 8.659873, 'Volksbank tower')
                }
    
    # Get your position from bearings to two stations
    triangulate(stations[1], 164.71, stations[3], 192.22).coordinates()
    
    # Get your position from bearings at least 3 stations (you can use more)
    #               position,      distance
    multilaterate([(stations[1],  1.275251),  
                   (stations[2],  1.599237),  
                   (stations[3],  1.917145)]).coordinates()
    