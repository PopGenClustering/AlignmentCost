from setuptools import setup, find_packages
import sys

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='AlignmentCost',
    version='0.0.1',
    author='Xiran Liu et al',
    zip_safe=False,
    packages=find_packages(),
    install_requires=["numpy","pandas","scipy","matplotlib","seaborn","argparse"],
    tests_require=[],
    include_package_data=True,
    long_description=long_description,
    long_description_content_type='text/markdown'
)   