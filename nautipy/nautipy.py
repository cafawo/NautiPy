# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 NautiPy Developers. All rights reserved.
Use of this source code is governed by the MIT license that can be found in
LICENSE.txt
"""

import numpy as np
from math import radians, cos, sin, asin, sqrt, degrees, atan2
from scipy.optimize import minimize
import json
import re


class Pos():
    """Class handling a single nautical (lat, lon) position
    """
    def __init__(self, lat: str or float, lon: str or float, 
                 desc:str=None, upid:str=None):
        """
        Parameters
        ----------
        lat : str or float
            ISO 6709 latitude: ±DD.D  (e.g. +50.12257).
        lon : str or float
            ISO 6709 longitude: ±DDD.D (e.g. +008.66370).
        desc : str, optional
            Description or name of the position. The default is None.
        upid : str, optional
            Unique position ID (upid). The default is None.
        """
        self.lat = lat
        self.lon = lon
        self.desc = desc
        self.upid = upid
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
            

def convert_coordinates(lat, lon, format: str = "DMS"):
    """
    Convert various GPS input formats to decimal degrees (ISO 6709).
    Supported formats:
    - Degrees, Minutes, Seconds (DMS): "N 50° 7' 20.9122"", "W 8° 39' 56.52""
    - Degrees and Decimal Minutes (DMM): "50° 7.34854N", "8° 39.942W"
    - Decimal Degrees (DD): "+50.12257", "-8.66570"
    
    Parameters:
    - lat (str or float): Latitude in various formats.
    - lon (str or float): Longitude in various formats.
    - format (str): Input coordinate format, one of "DMS", "DMM", or "DD".

    Returns:
    - tuple: (latitude, longitude) in decimal degrees (ISO 6709).
    """
    def dms_to_dd(d, m, s, direction):
        sign = -1 if direction.upper() in ['S', 'W'] else 1
        return sign * (d + m / 60 + s / 3600)
    
    def ddm_to_dd(d, m, direction):
        sign = -1 if direction.upper() in ['S', 'W'] else 1
        return sign * (d + m / 60)

    if format == "DD":
        return float(lat), float(lon)
    
    elif format == "DMS":
        lat_match = re.match(r"([NS])\s*(\d+)°\s*(\d+)'\s*(\d+\.?\d*)", lat.strip(), re.I)
        lon_match = re.match(r"([EW])\s*(\d+)°\s*(\d+)'\s*(\d+\.?\d*)", lon.strip(), re.I)
        if lat_match and lon_match:
            lat_dd = dms_to_dd(float(lat_match.group(2)), float(lat_match.group(3)), float(lat_match.group(4)), lat_match.group(1))
            lon_dd = dms_to_dd(float(lon_match.group(2)), float(lon_match.group(3)), float(lon_match.group(4)), lon_match.group(1))
            return lat_dd, lon_dd
    
    elif format == "DMM":
        lat_match = re.match(r"(\d+)°\s*(\d+\.?\d*)([NS])", lat.strip(), re.I)
        lon_match = re.match(r"(\d+)°\s*(\d+\.?\d*)([EW])", lon.strip(), re.I)
        if lat_match and lon_match:
            lat_dd = ddm_to_dd(float(lat_match.group(1)), float(lat_match.group(2)), lat_match.group(3))
            lon_dd = ddm_to_dd(float(lon_match.group(1)), float(lon_match.group(2)), lon_match.group(3))
            return lat_dd, lon_dd
        else:
            raise ValueError("Invalid DMM format.")
    
    else:
        raise ValueError("Unsupported coordinate format. Choose 'DD', 'DMS', or 'DMM'.")


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


def nearest_stations(pos:Pos, stations:list, radius:float=np.inf) -> list:
    """Return nearest stations to a given position
    
    Parameters
    ----------
    pos : TYPE
        Center position.
    stations : list
        List of Pos objects.
    radius : float, optional
        Only show stations within radius. The default is np.inf.

    Returns
    -------
    list
        List of nearest stations sorted by distance (ASC).

    """
    distances = [haversine(pos, s) for s in stations]
    stations = np.array(list(zip(stations, distances)))
    stations = stations[stations[:,1].argsort()]
    return stations[stations[:,1] <= radius]


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


def export_positions(positions, format: str = "geojson", filepath: str = None):
    """Export positions to a specified format and optionally save to a file.

    Parameters
    ----------
    positions : list
        List of Pos objects.
    format : str, optional
        Export format, currently supports 'geojson'. The default is "geojson".
    filepath : str, optional
        File name to save the exported data. If None, data is returned instead.

    Returns
    -------
    str
        The exported data in the specified format if not saved to a file.
    """
    if format == "geojson":
        features = []
        for pos in positions:
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [pos.lon, pos.lat]
                },
                "properties": {
                    "desc": pos.desc,
                    "upid": pos.upid
                }
            })

        geojson = {
            "type": "FeatureCollection",
            "features": features
        }

        geojson_str = json.dumps(geojson, indent=2)

        if filepath:
            with open(filepath, "w", encoding="utf-8") as file:
                file.write(geojson_str)
            return f"GeoJSON data saved to {filepath}"
        else:
            return geojson_str
    else:
        raise ValueError(f"Unsupported format: {format}")
    

def import_positions(filepath: str) -> list:
    """
    Load GeoJSON file and convert it to a list of Pos objects.

    Parameters
    ----------
    filepath : str
        Path to the GeoJSON file.

    Returns
    -------
    list
        List of Pos objects.
    """
    with open(filepath, 'r') as file:
        geojson_data = json.load(file)

    positions = []
    for feature in geojson_data['features']:
        lon, lat = feature['geometry']['coordinates']
        description = feature['properties'].get('desc', None)
        upid = feature['properties'].get('upid', None)
        positions.append(Pos(lat, lon, desc=description, upid=upid))

    return positions

    
#%% README.md
if __name__ == '__main__':
    ### Basics
    # Store and describe your position
    work = Pos(50.127198, 8.665562, desc='Campus building')
    print(f'Work is here: {work.coordinates()}')

    # Convert different formats to decimal degrees (ISO 6709)
    print("DD:  Latitude: {:.4f}, Longitude: {:.4f}".format(*convert_coordinates("+50.12257", "-8.66570", "DD")))
    print("DMS: Latitude: {:.4f}, Longitude: {:.4f}".format(*convert_coordinates("N 50° 7' 20.9122", "W 8° 39' 56.52", "DMS")))
    print("DMM: Latitude: {:.4f}, Longitude: {:.4f}".format(*convert_coordinates("50° 7.34854N", "8° 39.942W", "DMM")))
    
    # Get relative position heading 90 degrees 12 kilometers away
    work_displaced = work.displace(90, 12)
    
    # Get bearing to position
    bearing(work, work_displaced)
    
    # Get distance to position
    haversine(work, work_displaced)
    
    # Consider the following known stations
    stations = [Pos(50.116135, 8.670277, 'Opernturm'),
                Pos(50.112836, 8.666753, 'Deka tower'),
                Pos(50.110347, 8.659873, 'Volksbank tower')
                ]
    
    # Get the nearest stations within a radius around you position
    nearest = nearest_stations(work, stations, radius=1.7)
    print(f'{[(p.desc, d) for p, d in nearest]}')
    
    # Get your position from bearings to two stations
    triangulate(stations[0], 164.71, stations[2], 192.22).coordinates()
    
    # Get your position from bearings at least 3 stations (you can use more)
    #               position,      distance
    multilaterate([(stations[0],  1.275251),  
                   (stations[1],  1.599237),  
                   (stations[2],  1.917145)]).coordinates()
    
    # Export your positions to a geojson file
    print(export_positions(stations, filepath="example.geojson"))

    # Load positions
    stations_reloaded = import_positions(filepath="example.geojson")
    
    