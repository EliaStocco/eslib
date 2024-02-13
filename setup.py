from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="eslib",
    version="0.0.0",
    description="Some scripts",
    #package_dir={"": "miscellaneous"},
    packages=find_packages(), # where="miscellaneous"
    long_description=long_description,
    #long_description_content_type="text/markdown",
    #url="git@github.com:EliaStocco/miscellaneous.git",
    author="Elia Stocco",
    #author_email="stocco@fhi-berlin.mpg.de",
    # license="MIT",
    # classifiers=[
    #     "License :: OSI Approved :: MIT License",
    #     "Programming Language :: Python :: 3.10",
    #     "Operating System :: OS Independent",
    # ],
    # install_requires=["bson >= 0.5.10"],
    # extras_require={
    #     "dev": ["pytest>=7.0", "twine>=4.0.2"],
    # },
    # python_requires=">=3.10",
)
