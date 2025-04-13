from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name = "smPy",
    version = "0.0.4",
    description = "An all-in-one software package for the analysis and visualisation of single molecule images",
    package_dir= {"" : "src"},
    packages = find_packages(where="src"),
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url= "https://github.com/ljhason/smPy",
    author = "Lily Anderson", 
    author_email="lily.jian.hua.anderson@gmail.com",
    license = "MIT",
    classifiers = [
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires = [
        "numpy>=1.21",
        "opencv-python>=4.5",
        "matplotlib>=3.5",
        "scikit-image>=0.18",
        "Pillow>=8.0",  
    ],
    extras_require = {
        "dev": ["twine>=4.0.2"],
},
    python_requires = ">=3.6"
)