from setuptools import setup, find_packages


setup(name='nonechucks',
      version='0.2.1',
      url='https://github.com/msamogh/nonechucks',
      license='MIT',
      author='Amogh Mannekote',
      author_email='msamogh@gmail.com',
      description="""nonechucks is a library for PyTorch that provides wrappers for PyTorch's Dataset and Sampler objects to allow for dropping unwanted or invalid samples dynamically during dataset iteration.""",
      packages=find_packages(),
      long_description=open('README.md').read(),
      long_description_content_type='text/markdown',
      zip_safe=False)
