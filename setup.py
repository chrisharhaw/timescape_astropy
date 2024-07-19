from setuptools import setup, find_packages

setup(
    name='timescape',
    version='1.0.0',
    author='Christopher Harvey-Hawes',
    author_email='christopher.harvey-hawes@pg.canterbury.ac.nz',
    description='A package for using the Timescape cosmology',
    packages=find_packages(),
    scripts=['timescape.py'],
    install_requires=[
        'numpy',
        'scipy', 
        'astropy', 
        'matplotlib'
        'scipy',
        'numbers',
        'math',
        'astropy'],
    )