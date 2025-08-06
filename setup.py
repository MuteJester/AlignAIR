from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='AlignAIR',
    version='2.0.1',
    author='Thomas Konstantinovsky & Ayelet Peres',
    author_email='thomaskon90@gmail.com',
    description='Unified Multi-Chain IG/TCR Sequence Alignment Tool with Dynamic GenAIRR Integration',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/MuteJester/AlignAIR',
    project_urls={
        "Bug Tracker": "https://github.com/MuteJester/AlignAIR/issues"
    },
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    include_package_data=True,
    install_requires=requirements,
    license="GNU General Public License v3.0 or later (GPLv3+)",
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
    python_requires='>=3.9, <3.12',
    entry_points={
        'console_scripts': [
            'alignair_predict=AlignAIR.API.AlignAIRRPredict:main',
        ],
    },
)
