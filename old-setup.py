import os
import subprocess
from setuptools import setup, find_packages
from setuptools.command.install import install

# Read the long description from README.md
with open("README.md", 'r', encoding="utf-8") as f:
    long_description = f.read()

# Define a custom install command to run post-install actions
class PostInstallCommand(install):
    """Custom installation for setting environment variables and making Python scripts executable."""
    def run(self):
        install.run(self)  # First, proceed with the standard installation

        # Custom post-install script logic to mimic shell script functionality
        eslib_dir = os.path.dirname(os.path.abspath(__file__))

        # Add eslib and subdirectories to PATH and PYTHONPATH
        os.environ["PATH"] += f":{eslib_dir}/eslib"
        os.environ["PYTHONPATH"] += f":{eslib_dir}/eslib"

        for dirpath, _, filenames in os.walk(f"{eslib_dir}/eslib/scripts"):
            os.environ["PATH"] += f":{dirpath}"
            for filename in filenames:
                if filename.endswith(".py"):
                    file_path = os.path.join(dirpath, filename)
                    os.chmod(file_path, 0o755)  # Make Python scripts executable

        cluster_dir = os.path.join(eslib_dir, "cluster")
        os.environ["PATH"] += f":{cluster_dir}"
        os.environ["PYTHONPATH"] += f":{cluster_dir}"

        fortran_dir = os.path.join(eslib_dir, "eslib", "eslib", "fortran")
        os.environ["PYTHONPATH"] += f":{fortran_dir}"

        # Optionally source archive.sh or execute it
        archive_sh = os.path.join(cluster_dir, "archive.sh")
        if os.path.exists(archive_sh):
            subprocess.run(["bash", archive_sh])

# Define the setup configuration
setup(
    name="eslib",
    version="0.0.1",  # Increment the version as needed
    description="Some scripts",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="git@github.com:EliaStocco/miscellaneous.git",
    author="Elia Stocco",
    author_email="stocco@fhi-berlin.mpg.de",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),  # Automatically find and include all packages
    python_requires=">=3.9",
    install_requires=[
        # Uncomment and add the required packages as needed
        # "bson >= 0.5.10",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "twine>=4.0.2"],
    },
    setup_requires=['wheel'],
    cmdclass={
        'install': PostInstallCommand,  # Hook the post-install function
    },
)
