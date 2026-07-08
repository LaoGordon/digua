import os
from glob import glob

from setuptools import find_packages, setup


package_name = "fire_pitch_aim_ros2"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="sunrise",
    maintainer_email="sunrise@todo.todo",
    description="Lite3 pitch aiming based on fire detection vertical image error.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "fire_pitch_aim_node = fire_pitch_aim_ros2.fire_pitch_aim_node:main",
        ],
    },
)
