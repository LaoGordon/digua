from setuptools import setup

package_name = "patrol2_ros2"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", ["launch/patrol2.launch.py"]),
        (f"share/{package_name}/config", ["config/patrol2_params.yaml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="sunrise",
    maintainer_email="sunrise@localhost",
    description="Simplified fixed-waypoint patrol that reuses Nav2, YOLO, and pitch aim.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "patrol2_node = patrol2_ros2.patrol2_node:main",
        ],
    },
)
