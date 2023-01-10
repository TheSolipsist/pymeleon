import setuptools

# Developer self-reminder for uploading in pypi:
# - install: wheel, twine
# - build  : python setup.py bdist_wheel
# - deploy : twine upload dist/*

with open("README.md", "r") as file:
    long_description = file.read()

setuptools.setup(
    name='pymeleon',
    version='0.1.0',
    author="Orestis Farmakis, Emmanouil (Manios) Krasanakis",
    author_email="",  # set your email here and in the readme
    description="Runtime type-driven synthesis based on domain specific language transformations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TheSolipsist/pymeleon",
    packages=setuptools.find_packages(),
    classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: Apache Software License",
         "Operating System :: OS Independent",
     ],
    install_requires=[
              'scipy', 'torch', 'networkx'  # add missing requiements here
    ],
 )
