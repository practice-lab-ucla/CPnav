from setuptools import setup, find_packages

setup(
    name="CPnav",
    version="0.1",
    packages=find_packages(),  # this will pick up "CPnav" automatically
    install_requires=[
        # put your dependencies here if you want them auto-installed
        "numpy",
        # "habitat-sim", "magnum", etc, if needed
    ],
)
