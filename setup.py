import pathlib
import setuptools

# The directory containing this file
HERE = pathlib.Path(__file__).parent
# The text of the README file
README = (HERE / "README.md").read_text()
# This call to setup() does all the work
setuptools.setup(
    name="torch_lydorn",
    version="0.0.2",
    description="Various extensions to existing PyTorch libraries",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Nicolas Girard",
    author_email="nicolas.jp.girard@gmail.com",
    license="BSD-3-Clause license",
    classifiers=[
        "License :: BSD-3-Clause license",
        "Programming Language :: Python"
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.9"
)
