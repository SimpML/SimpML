import os
from setuptools import setup, find_packages

# Helper function to read requirements from a file
def parse_requirements(filename):
    with open(os.path.join(os.path.dirname(__file__), filename), 'r') as file:
        return [line.strip() for line in file if line.strip() and not line.startswith('#')]

# Read main dependencies and dev dependencies
install_requires = parse_requirements('requirements.txt')
dev_requires = parse_requirements('dev-requirements.txt')

setup(
    name="simpml",
    version="0.1",
    description="SimpML is an open-source, no/low-code machine learning library in Python that simplifies and automates machine learning workflows.",
    author="Miriam Horovicz, Roni Goldschmidt",
    author_email="miryam.hor@gmail.com, ronigoldsmid@gmail.com",
    packages=find_packages(include=["simpml", "simpml.*"]),
    install_requires=install_requires,  # Main dependencies
    extras_require={
        'dev': dev_requires,  # Development dependencies
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8.1,<3.12",
)
