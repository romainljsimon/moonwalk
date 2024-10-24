from setuptools import setup, find_packages

setup(
    name='sphere_walk',
    version='0.1',
    packages=find_packages(),
    install_requires=[],  # List dependencies here, if any
    entry_points={
        'console_scripts': [
            'sphere_walk=sphere_walk.run:main'
        ],
    },
)
