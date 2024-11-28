from setuptools import find_packages, setup

import versioneer


# Function to parse requirements.txt
def parse_requirements(filename):
    with open(filename, "r") as f:
        return f.read().splitlines()


setup(
    name="onenet",  # Name of your package
    version=versioneer.get_version(),  # Initial release version
    cmdclass=versioneer.get_cmdclass(),  # Command line version
    packages=find_packages(),  # Automatically find and include all packages
)
