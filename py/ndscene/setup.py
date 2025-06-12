from setuptools import setup, find_packages

setup(
    name='ndscene',
    version='0.1',
    packages=find_packages(exclude=['tests*']),
    license='MIT',
    description='n-dimensional graphics library',
    long_description=open('README.txt').read(),
    install_requires=['numpy'],
    url='https://github.com/leweyg/ndscene',
    author='Lewey Geselowitz',
    author_email='leweygeselowitz@gmail.com'
)