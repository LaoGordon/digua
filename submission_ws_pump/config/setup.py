from setuptools import setup

package_name = "pump_ros2"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", ["config/pump_params.yaml"]),
        (f"share/{package_name}/launch", ["launch/pump_spray.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="sunrise",
    maintainer_email="sunrise@localhost",
    description="Water pump control: hold pin LOW, spray N sec on /pump/cmd topic.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "pump_node = pump_ros2.pump_node:main",
            "pump_spray_node = pump_ros2.pump_spray_node:main",
        ],
    },
)
