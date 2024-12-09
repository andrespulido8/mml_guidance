#!/bin/bash

seconds=100
N_sims=2

roslaunch mml_guidance mml_sim_estimator.launch &
sleep $seconds

# Loop to launch the second ROS launch file N_sims times
for ((i=1; i<=N_sims; i++))
do
    echo "Iteration $i"
    rostopic pub -1 /robot0/goal_pose geometry_msgs/Pose "{
        position: {
            x: -1.25,
            y: -0.6,
            z: 0.0
        },
        orientation: {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 1.0
        }
    }"
    sleep 5
    roslaunch mml_guidance sim_experiments.launch &
    sleep $seconds
done
echo "DONE"