from setuptools import setup, find_packages


setup(
    name="scbd",
    packages=find_packages(),
    install_requires=[
        "boto3",
        "lightning",
        "lmdb",
        "matplotlib",
        "pandas",
        "pyarrow",
        "seaborn",
        "scikit-learn",
        "torch",
        "torchvision",
        "wilds"
    ]
)