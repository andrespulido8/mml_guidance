#!/bin/bash

seconds=190  # this should be longer than the sim shutdown time
N_sims=10

roslaunch mml_guidance mml_sim_estimator.launch &
roslaunch mml_guidance bag_sim.launch prefix_name:="KF" bag_selected:="true" &
sleep $seconds

# Loop to launch the second ROS launch file N_sims times
for ((i=1; i<=N_sims-1; i++))
do
    echo "Iteration $i"
    rostopic pub -1 /robot0/goal_pose geometry_msgs/Pose "{
        position: {
            x: 0.0,
            y: 0.0,
            z: 0.0
        }, orientation: {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 1.0
        }
    }"
    sleep 5
    roslaunch mml_guidance sim_experiments.launch &
    roslaunch mml_guidance bag_sim.launch prefix_name:="PFVelocity" bag_selected:="true" &
    sleep $seconds
done
echo "killing processes"
pkill record
pkill roslaunch
pkill python
pkill reef_estimator
pkill reef_control
sleep 5

echo "DONE WITH REPEATED SIMS"
