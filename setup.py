from setuptools import setup, find_packages

setup(
    name="movie_recommender_system",
    version="0.1",
    author="Konstantinos Nikoletos",
    author_email="cs22000222@di.uoa.gr",
    # description="A brief description of your package",
    # long_description="A longer, more detailed description of your package",
    packages=find_packages(),
    install_requires=[
        "numpy", "pandas", "scikit-learn"
    ],
)