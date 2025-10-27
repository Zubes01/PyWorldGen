# PyWorldGen

My fantasy world generator I built using python. Work in progress!

### Features:

- The generation of (mostly!) hexagon tiled spherical worlds 
    - Using Goldberg's method 
    - All hexagons except for 12 equidistant pentagons
    - Configurable resolution level
    - Caching to avoid recomputation
- OpenSimplex based terrain generation for tile-based spherical worlds
    - Configurable scale and number of octaves
- Plot your world map as an equirectangular Projection
- Render your generated worlds with vispy
    - Interactive
    - Multiple lighting options
    - Metal-supported
    - Tile heights can be rendered

### Coming soon:

- Plate tectonics based world generation