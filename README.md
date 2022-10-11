# mml_guidance

Motion Model Learning (MML) guidance algorithms for turtlebot and quad-copter

## Requirements
- [Reef estimator simulation](https://github.com/uf-reef-avl/reef_estimator_sim_bundle)
- [Turtlebot packages](https://automaticaddison.com/how-to-launch-the-turtlebot3-simulation-with-ros/#gazebo)
- mag_pf and mag_pf_visualization

## Sim Usage

- To adjust the number of simulated vehicles modify the `./launch/launch_sim.launch` file inside the sim_helper repository.
- Change the `spawn_turtles` argument inside the previously mentioned launch file to `robot0`.
- Wait a few seconds until __Autopilot ARMED__ and __RC override active__ are printed and then in another terminal run `roslaunch mml mml_sim_estimator.launch` from the launch directory.

## Visualization
To visualize the particle filter and the motion model, run `roslaunch mml visualization.launch`.

## Train
- To turn off the Gazebo GUI to make the sim faster, change the argument `gui` to `false` in `camera_multirotor.launch` inside the **sim_helper** package from REEF github

## Hardware Usage
To run only the turtlebot, do `roslaunch mml turtlebot_hardware.launch`.

## Contributing Guide
To make changes to this repo, it is recommended to use the tool [pre-commit](https://pre-commit.com/).
To install it, run `pip3 install -r requirements.txt` inside this repo, and then install the hooks
specified in the config file by doing `pre-commit install`. Now to run it against all the files to check
if it worked, run `pre-commit run --all-files`.

## Profiling
Run `roslaunch mml mml_sim_estimator.launch` and then `pprofile --format callgrind --out guidance.pprofile /home/andrespulido/catkin_ws/src/mml_guidance/scripts/guidance.py __name:=drone_guidance`. This will run the profiler and save the results in the directory where the command was called.