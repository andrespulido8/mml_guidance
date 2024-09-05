from setuptools import find_packages, setup
from glob import glob
package_name = 'mml_guidance'

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.xml")),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
        ("share/" + package_name + "/config", glob("config/*.config.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="andres pulido",
    maintainer_email="andrespulido@ufl.edu",
    description="Launch package for mml guidance project",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": ["guidance = mml_guidance.guidance:main",
                            "markov_goal_pose= mml_guidance.markov_goal_pose:main", 
                            "mml_pf_visualization = mml_guidance.mml_pf_visualization:main", 
        ]
    },
    
)