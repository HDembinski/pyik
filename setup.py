import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyik",
    version="0.9",
    author="Hans Dembinski",
    author_email="hans.dembinski@gmail.com",
    description="PyIK - The Python Instrument Kit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="BSD",
    url="https://github.com/hdembinski/pyik",
    packages=setuptools.find_packages(),
    tests_require=["numpy", "scipy", "matplotlib"],
    classifiers=[
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
)
