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
    ],
    python_requires=">=3.8",
)
