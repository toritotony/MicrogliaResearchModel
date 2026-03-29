import setuptools
setuptools.setup(     
     name="MicrogliaResearchModel",     
     version="1.0.0",
     author='Anthony Wolfe',
     author_email='wolfeanthony584@gmail.com',
     description='An Agent Based Model simulating immune cell interactions in CNS.',
     long_description=open('README.md').read(),
     long_description_content_type='text/markdown',
     install_requires=[
          'mesa',
          'numpy',
          'matplotlib',
          'pandas',
          'scipy',
          'seaborn',
     ],
     python_requires=">=3.12",   
     packages=["MicrogliaResearchModel"],
)