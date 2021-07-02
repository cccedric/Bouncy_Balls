# Bouncy_Balls

This project simulate a system of bouncing balls. The balls will have mass and collide semi-elastically with each other and the walls of a box.

This simulation is conducted via a timestep approach. Every tcol milliseconds (configurable in config.py), the simulation will check for collisions that balls have with the walls and with each other, and will alter the trajectory of balls accordingly.