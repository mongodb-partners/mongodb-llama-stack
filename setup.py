from setuptools import setup, find_packages

# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

setup(
    name="mongodb-llama-stack",
    version="0.1.0",
    description="MongoDB Atlas Vector Search provider for Llama Stack with hybrid search capabilities",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="MongoDB Partners",
    author_email="partners@mongodb.com",
    url="https://github.com/mongodb-partners/llama-stack-provider-mongodb",
    packages=find_packages(include=["mongodb_llama_stack", "mongodb_llama_stack.*"]),
    python_requires=">=3.10",
    install_requires=[
        "llama-stack>=0.0.53",
        "pymongo>=4.5.0",
        "certifi>=2023.7.22",
        "numpy>=1.24.0",
        "packaging>=23.0",
        "pydantic>=2.0.0",
    ],
    extras_require={
        "test": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-mock>=3.10.0",
        ],
        "dev": [
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.0.0",
        ],
    },
    package_data={
        "mongodb_llama_stack": [
            "README.md",
            "providers.d/remote/vector_io/mongodb.yaml",
            "*.yaml",
            "*.yml",
        ],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Database",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords=[
        "mongodb",
        "vector-search",
        "llama-stack",
        "ai",
        "machine-learning",
        "semantic-search",
        "hybrid-search",
        "atlas-search",
        "embeddings",
    ],
    entry_points={
        "console_scripts": [
            "mongodb-llama-demo=examples.demo:main",
        ],
    },
)