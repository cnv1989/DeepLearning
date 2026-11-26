"""drone: a quadcopter simulator built from scratch, for learning how drones fly.

Modules, in the order a drone's software uses them:

  rotations   Quaternions, rotation matrices and Euler angles (frames)
  params      The physical description of the quadcopter
  motors      Motors, propellers and the mixer
  battery     Battery voltage and charge
  dynamics    Rigid-body physics: forces, torques, motion (the "real world")
  sensors     IMU, barometer, GPS and magnetometer, with realistic noise
  estimation  Turning noisy sensor readings into a best guess of the state
  control     PID controllers, cascaded into a full flight controller
  planning    A* path planning and waypoint following
  mission     Flight modes: take off, fly the mission, land
  simulation  The loop that ties everything together
  plots       Charts and 3D views of a flight
"""
