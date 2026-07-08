import os
from glob import glob

from setuptools import find_packages, setup


package_name = "fire_yolo_bpu_ros2"


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
    description="RDK BPU fire detection with RealSense streaming, on-screen display, and 3D localization publishing.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "fire_yolo_bpu_node = fire_yolo_bpu_ros2.fire_yolo_bpu_node:main",
        ],
    },
)
