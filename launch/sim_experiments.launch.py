import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition

def generate_launch_description():
    # Declare the argument for bag name
    declare_bag_name_arg = DeclareLaunchArgument(
        'bag_name',
        default_value='mml_guidance',
        description='Name of the bag to use.'
    )

    # Declare the argument for simulation mode
    declare_is_sim_arg = DeclareLaunchArgument(
        'is_sim',
        default_value='true',
        description='Whether to run in simulation mode.'
    )

    # launch markov_goal_pose node
    markov_goal_pose_node = Node(
        package='mml_guidance',
        executable='markov_goal_pose',
        name='markov_goal_pose',
        output='screen',
        remappings=[
            ('/goal_pose', '/leo/goal_pose'),
            ('/robot0/odom', '/leo/odom'),
        ],
        parameters=[
            {'is_sim': LaunchConfiguration('is_sim')},
        ]
    )

    # Launch guidance node
    guidance_node = Node(
        package='mml_guidance',
        executable='guidance',
        name='drone_guidance',
        output='screen',
        remappings=[
            ('/turtle_pose_stamped', '/leo/enu/pose'),
            ('/quad_pose_stamped', '/pop/enu/pose')
        ],
        parameters=[
            {'is_viz': True},
            {'is_sim': LaunchConfiguration('is_sim')}
        ]
    )

    # Launch mml_pf_visualization node
    mml_pf_visualization_node = Node(
        package='mml_guidance',
        executable='mml_pf_visualization',
        name='mml_pf_visualization',
        output='screen',
        remappings=[
            ('odom', '/leo/enu/pose')
        ],
        parameters=[
            {'is_sim': LaunchConfiguration('is_sim')}
        ]
    )

    return LaunchDescription([
        declare_bag_name_arg,
        declare_is_sim_arg,
        guidance_node,
        mml_pf_visualization_node,
        markov_goal_pose_node
    ])