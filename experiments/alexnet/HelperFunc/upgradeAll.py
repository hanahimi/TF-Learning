#!/usr/bin/env python
"""
    Upgrade all python packages installed in this computer
"""
import pip
from subprocess import call

for dist in pip.get_installed_distributions():
  call("pip install --upgrade " + dist.project_name, shell=True)
