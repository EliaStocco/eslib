import os
import subprocess
from setuptools import setup, find_packages
from setuptools.command.develop import develop

class CustomDevelopCommand(develop):
    """Custom develop command to run compile.sh scripts."""
    def run(self):
        for folder in ["eslib/fortran/rdf", "eslib/fortran/msd"]:
            script_path = os.path.join(folder, "compile.sh")
            if os.path.isfile(script_path):
                print(f"Running {script_path} ...")
                # Run 'bash compile.sh' inside folder, so script_path inside folder is just 'compile.sh'
                subprocess.check_call(["bash", "compile.sh"], cwd=folder)
            else:
                print(f"Warning: {script_path} not found.")
        super().run()

setup(
    name="eslib",
    version="1.0.0",
    description="ESLIB installation and setup.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Elia Stocco",
    author_email="eliastoccol@gmail.com",
    license="MIT",
    keywords=["materials", "science", "ML"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "colorama",
        "ase",
        "chemiscope",
        "icecream",
        "matplotlib",
        "matscipy",
        "pandas",
        "pint",
        "scikit-learn",
        "scipy>=1.2.3",
        "seekpath",
        "skmatter",
        "spglib",
        "tqdm",
        "xarray",
        "pytest",
        "phonopy",
        "uncertainties",
        "meson",
        "ninja"
    ],
    extras_require={
        "mace": ["mace-torch"],
        "dev": ["pytest"],
    },
    packages=find_packages(include=["eslib*"]),
    include_package_data=True,
    zip_safe=False,
    cmdclass={
        'develop': CustomDevelopCommand,
    },
)
