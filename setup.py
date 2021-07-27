
from setuptools import setup, find_packages

setup(
    name='fastai_object_detection',
    version='0.0.1',
    url='https://github.com/rbrtwlz/fastai_object_detection.git',
    author='Robert Walz',
    author_email='abitdevelopment@gmail.com',
    description='fastai extension for object detection',
    packages=find_packages(),    
    install_requires = ['torch==1.7.1', 'torchvision==0.8.2', 'fastai', 'mean_average_precision @ git+https://github.com/bes-dev/mean_average_precision', 'pycocotools']
)
