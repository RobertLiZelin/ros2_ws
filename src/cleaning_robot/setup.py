from setuptools import setup

package_name = 'cleaning_robot'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='axt',
    maintainer_email='axt@todo.todo',
    description='Room cleaning algorithm with A* + neighbor-first traversal in ROS 2 Jazzy',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'cleaner_node = cleaning_robot.cleaner_node:main',
        ],
    },
)

