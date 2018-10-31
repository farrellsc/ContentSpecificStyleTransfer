from setuptools import setup, find_packages
import codecs


def read_install_requirements():
    with codecs.open('./requirements.txt', 'r', encoding='utf8') as f:
        content = f.readlines()
        requirements = [line.strip('\n') for line in content]
    return requirements


setup(
    name="SuperStyleTransfer",
    version="0.0",
    description='',
    keywords='',
    url='https://github.com/farrellsc/SuperStyleTransfer',
    author='Xu Han, Ruolin Jia, Zhou Zhuang',
    license='MIT',
    packages=find_packages(),
    install_requires=read_install_requirements(),
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    include_package_data=True,
    python_requires='>=3.6'
)
