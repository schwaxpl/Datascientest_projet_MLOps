from setuptools import setup, find_packages

setup(
    name="mlops-project",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "mlflow",
        "fastapi",
        "pytest",
        "python-multipart",
        "uvicorn",
        "tensorflow",
        "keras"
    ]
)
