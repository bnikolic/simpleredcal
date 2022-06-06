from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'Generalized Approach to Redundant Calibration with JAX'
LONG_DESCRIPTION = 'Generalized Approach to Redundant Calibration with JAX'

# Setting up
setup(
        name='simpleredcal',
        version=VERSION,
        author='Matyas Molnar',
        author_email='mdm49@cam.ac.uk',
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[],
        keywords=['python', 'redundant', 'calibration', 'robust']
)
