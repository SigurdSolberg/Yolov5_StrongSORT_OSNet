from setuptools import setup

with open("README", 'r') as f:
    long_description = f.read()

setup(
   name='yolo_strongsort',
   version='1.0',
   description='A useful module',
   license="MIT",
   long_description=long_description,
   author='Man Foo',
   author_email='foomail@foo.example',
   url="https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet",
   packages=['yolo_strongsort'],  #same as name
   install_requires=['wheel'], #external packages as dependencies
)