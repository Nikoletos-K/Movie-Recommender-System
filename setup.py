from setuptools import setup, find_packages

setup(
    name="movie_recommender_system",
    version="0.1",
    author="Konstantinos Nikoletos",
    author_email="cs22000222@di.uoa.gr",
    packages=find_packages(),
    install_requires=[
        "numpy", "pandas", "tqdm", "scipy"
    ],
)