# NautiPy
NautiPy is an open-source Python library for nautical navigation applications.


## ISO 6709 Coordinates
The library uses the ISO 6709 standard for GPS coordinates, i.e.
* Latitude: ±DD.D  (e.g. +50.12257)
* Longitude: ±DDD.D (e.g. +008.66370)

Future versions will include a conversion function from other formats.


## Functionalities

### Basics
```
# Store and descibe your position
work = Pos(50.127198, 8.665562, desc='Campus building')
print(f'I work at {work.coordinates()}')

# Get relative position heading 90 degrees 12 kilometers away
work_displaced = work.displace(90, 12)

# Get bearing to position
bearing(work, work_displaced)

# Get distance to position
haversine(work, work_displaced)
```

### Triangulation and Multilateration
To fix a position both methods rely on knowledge about the positon and bearing to 2 (triangulation) or position and distance to 3 (multilateration) stations.
```
# Save some known locations (stations)
stations = {1:Pos(50.116135, 8.670277, 'Opernturm'),
            2:Pos(50.112836, 8.666753, 'Deka tower'),
            3:Pos(50.110347, 8.659873, 'Volksbank tower')
            }
```
In trigonometry and geometry, triangulation is the process of determining the location of a point by forming triangles to it from known points. ([Wikipedia](https://en.wikipedia.org/wiki/Triangulation))
```
# Get your position from bearings to two stations
triangulate(stations[1], 164.71, stations[3], 192.22)
```
True range multilateration is a method to determine the location of a movable vehicle or stationary point in space using multiple ranges (stations) between the vehicle/point and multiple spatially-separated known locations. ([Wikipedia](https://en.wikipedia.org/wiki/Triangulation))
```
# Get your position from bearings at least 3 stations (you can use more)
#               position,      distance
multilaterate([(stations[1],  1.275251),  
               (stations[2],  1.599237),  
               (stations[3],  1.917145)])
```
Compare both positions to work, i.e. (50.127198, 8.665562).
