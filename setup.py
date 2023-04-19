from setuptools import find_packages,setup
from typing import List
import os


HYPEN_E_DOT='-e .'


def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as f:
        requirements=f.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name='Regression_project_Delivery_time',
    version='0.1',
    description='This package is for Predicting delivery time on the basis of old data in regression ML model',
    author='Aaron17',
    author_email='aaronnb17@gmail.com',
    install_requires=get_requirements(r"D:\ineuron\practice\ML_projects\project_tasks\deliverytime_ML_model\requirements.txt"),
    packages=find_packages()             
    )



