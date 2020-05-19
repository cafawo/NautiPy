# NautiPy
NautiPy is an open-source Python library for nautical navigation applications.


## ISO 6709 Coordinates
The library uses the ISO 6709 standard for GPS coordinates, i.e.
* Latitude: ±DD.D  (e.g. +50.12257)
* Longitude: ±DDD.D (e.g. +008.66370)

Future versions will include a conversion function from other formats.


## Functionalities

### Basics
```Python
# Store and descibe your position
work = Pos(50.127198, 8.665562, desc='Campus building')
print(f'Work is here: {work.coordinates()}')

# Get relative position, e.g. heading 90 degrees 12 kilometers away
work_displaced = work.displace(90, 12)

# Get bearing to or from position
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
```

### Triangulation and Multilateration
To fix a position both methods rely on knowledge about the position and bearing to 2 (triangulation) or position and distance to 3 (multilateration) stations.

In trigonometry and geometry, triangulation is the process of determining the location of a point by forming triangles to it from known points. ([Wikipedia](https://en.wikipedia.org/wiki/Triangulation))
```Python
# Get your position from bearings to two stations
triangulate(stations[0], 164.71, stations[2], 192.22).coordinates()
```
True range multilateration is a method to determine the location of a movable vehicle or stationary point in space using multiple ranges (stations) between the vehicle/point and multiple spatially-separated known locations. ([Wikipedia](https://en.wikipedia.org/wiki/Multilateration))
```Python
# Get your position from bearings at least 3 stations (you can use more)
#               position,      distance
multilaterate([(stations[0],  1.275251),  
               (stations[1],  1.599237),  
               (stations[2],  1.917145)]).coordinates()
```
Compare both positions to work, i.e. (50.127198, 8.665562).
