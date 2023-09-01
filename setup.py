from setuptools import setup
import pyxc

setup(
    name=pyxc.__name__,
    version=pyxc.__version__,
    packages=["pyxc", "pyxc.core", "pyxc.transform"],
    url="https://pyxc.readthedocs.io/en/latest/",
    license="MIT",
    author=pyxc.__author__,
    author_email="",
    description=pyxc.__description__,
    python_requires=">=3.10",
    package_data={
        "pyxc": ["py.typed"],
        "": ["LICENSE", "README.md", "readthedocs.yaml"],
    },
)