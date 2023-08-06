from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="cordmap",
    version="0.0.1",
    packages=find_packages(),
    url="",
    license="",
    author="J. Marcus Hughes",
    author_email="mhughes@boulder.swri.edu",
    description="code to train CORD map product",
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    install_requires=[
        "goes-solar-retriever",
        "numpy",
        "pandas",
        "torch",
        "torchvision",
        "astropy",
        "sunpy",
        "opencv-python",
        "matplotlib",
        "humanfriendly",
        "dask",
        "tqdm",
        "datasets",
        "monai",
        "transformers",
        "zarr"
    ],
    extras_require={
        "test": ["pytest", "coverage", 'pytest-runner'],
        "docs": ["jupyter-book"],
    },
)