from setuptools import setup, find_packages

setup(
    name='Joana Oliveira Gonçalves',
    version='0.0.1',
    python_requires='>=3.7',
    package_dir={'': 'src'},
    packages=find_packages('src'),
    zip_safe=False,
    install_requires=['numpy', 'pandas', 'scipy'],
    author='Joana Gonçalves',
    author_email='joanaoliveira1000@gmail.com',
    description='Sistemas inteligentes',
    license='Apache License Version 2.0',
    keywords='',
)
