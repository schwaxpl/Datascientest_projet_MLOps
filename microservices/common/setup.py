from setuptools import setup, find_packages

setup(
    name="mlops-common",
    version="0.1.0",
    packages=find_packages(include=["*"]),
    package_dir={"": "."},
    include_package_data=True,
    install_requires=[
        # Les dépendances sont déjà installées via requirements.txt
    ],
)
