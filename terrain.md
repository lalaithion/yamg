# Terrain Generation

This document is a design doc and documentation for the algorithm which
generates terrain for these maps.

## Prior Work

- https://www.youtube.com/watch?v=eaXk97ujbPQ
- https://dood.al/test/island/
- https://mewo2.com/notes/terrain/
- http://www.hempuli.com/blogblog/archives/1699
- https://davidar.io/post/sim-glsl

## Lakes

1 1 1 1 1 1 1
1 5 5 5 5 5 1
1 5 2 3 4 5 1
1 5 3 4 5 5 1
1 5 4 5 5 1 1
1 1 1 1 1 1 1

1 1 1 1 1 1 1
1 5 5 5 5 5 1
1 5 2 3 2 5 1
1 5 3 4 5 5 1
1 5 4 5 5 1 1
1 1 1 1 1 1 1
