from setuptools import setup, find_packages

setup(
    name='nautipy',
    version='0.1',
    packages=find_packages(exclude=['nautipy/tests*']),
    license='MIT',
    description='Nautical navigation',
    long_description=open('README.md').read(),
    install_requires=['numpy', 'math', 'scipy'],
    url='https://github.com/cafawo/NautiPy',
    author='CFW',
    maintainer="NautiPy Developers",
    author_email='cfw@pm.me'
)
