from setuptools import setup, find_packages

setup(
    name="mongodb-llama-stack",
    version="0.1.0",
    description="MongoDB provider for Llama Stack",
    packages=find_packages(include=["mongodb_llama_stack", "mongodb_llama_stack.*"]),
    python_requires=">=3.10",
    install_requires=[
        # Add your dependencies here
    ],
    package_data={
        "mongodb_llama_stack": ["README.md"],
    },
)