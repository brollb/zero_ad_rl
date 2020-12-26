import os
from setuptools import setup

with open('requirements.txt', 'r') as f:
    dependencies = [ line.strip() for line in f.readlines() ]

setup(name='zero_ad_rl',
      version='0.0.1',
      description='Tools for RL exporations in 0 AD',
      author='Brian Broll',
      author_email='brian.broll@gmail.com',
      install_requires=dependencies,
      license='MIT',
      packages=['zero_ad_rl'],
      zip_safe=False)
