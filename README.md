# mml_guidance

Motion Model Learning (MML) guidance algorithms for turtlebot and quad-copter

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