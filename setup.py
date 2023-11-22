from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.readlines()
    
setup(
    name='CoLeCT_sim',
    version='0.1',
    packages=find_packages(),
    author='Lorenzo Busellato',
    description='Simulation package for the CoLeCT project',
    install_requires=requirements
)
