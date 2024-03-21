from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='AlignAIR',
    version='0.1.0',
    author='Thomas Konstantinovsky & Ayelet Peres',
    author_email='thomaskon90@gmail.com',
    description='IG Sequence Alignment Tool Based on Deep Convolutional Neural Networks',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/MuteJester/AlignAIR',
    project_urls={
        "Bug Tracker": "https://github.com/MuteJester/AlignAIR/issues"
    },
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    include_package_data=True,  # Include everything in source control
    install_requires=requirements,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    keywords='immunogenetics, sequence alignment, bioinformatics',
    python_requires='>=3.7',
)
