"""setup.py file for packaging negations."""

from setuptools import setup


with open('readme.md', 'r') as readme_file:
    readme = readme_file.read()


setup(
    name='negations',
    version='0.0.1',
    description='description here',
    long_description=readme,
    url='https://github.com/{url_here}',
    author='Your Name',
    author_email='your_email_here',
    keywords='keyword1, keyword2',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3.6',
        'License :: OSI Approved :: Apache Software License',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    license='Apache',
    packages=['negations'],
    package_dir={'': 'src'},
    scripts=[],
    install_requires=[],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    include_package_data=True,
    python_requires='>= 3.8',
    zip_safe=False)
