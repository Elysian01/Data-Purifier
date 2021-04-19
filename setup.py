from setuptools import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

# https://pypi.org/pypi?%3Aaction=list_classifiers
classifiers = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Libraries :: Python Modules',
    'Operating System :: Microsoft :: Windows :: Windows 10',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.7'
]

setup(name="data-purifier",
      version="0.0.4",
      description="A Python library for Automated Exploratory Data Analysis, Automated Data Cleaning and Automated Data Preprocessing For Machine Learning and Natural Language Processing in Python.",
      url="",
      long_description=open("PYPI_README.md", encoding='utf-8').read(),
      long_description_content_type='text/markdown',
      keywords="automated eda exploratory-data-analysis data-cleaning data-preprocessing",
      author="Abhishek Manilal Gupta",
      author_email="abhig0209@gmail.com",
      license="MIT",
      classifiers=classifiers,
      install_requires=required,
      packages=["datapurifier"]
      )

# for pypi => python setup.py sdist
# twine upload dist/*


# python setup.py bdist_wheel
# twine upload dist/*
# pip install dist/Autoeda-0.0.1-py3-none-any.whl
