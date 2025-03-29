from setuptools import setup, find_packages

setup(
    name="customer_support_ai",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "fastapi>=0.109.2",
        "uvicorn>=0.27.1",
        "pydantic>=2.6.1",
        "pydantic-settings>=2.1.0",
        "groq>=0.4.2",
        "chromadb>=0.4.22",
        "sentence-transformers>=2.5.1",
        "python-dotenv>=1.0.1",
        "pandas>=2.2.1",
        "numpy>=1.26.4",
        "scikit-learn>=1.4.1.post1",
        "python-multipart>=0.0.9",
        "datasets>=2.17.1",
        "tqdm>=4.66.2",
        "rank-bm25>=0.2.2",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-cov>=4.1.0",  # for coverage reporting
            "black>=24.2.0",  # for code formatting
            "isort>=5.13.2",  # for import sorting
            "flake8>=7.0.0",  # for linting
            "mypy>=1.8.0",  # for type checking
        ]
    },
    python_requires=">=3.8",
)
