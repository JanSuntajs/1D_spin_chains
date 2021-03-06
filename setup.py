from setuptools import setup, find_packages


setup(name='spins1d',
      version='1.0.0',
      description='A module for calculations with 1D spin chains',
      url='https://github.com/JanSuntajs/1D_spin_chains',
      author='Jan Suntajs',
      author_email='Jan.Suntajs@ijs.si',
      license='MIT',
      packages=find_packages(),
      install_requires=['spectral_stats', 'dataIO'],
      dependency_links=[('https://github.com/JanSuntajs/'
                         'spectral_statistics_tools/tarball/'
                         'master/#egg=spectral_stats-1.0.1'),
                        ('https://github.com/JanSuntajs/'
                         'spectral_statistics_tools/tarball/'
                         'master/#egg=dataIO-1.0.1')],
      zip_safe=False)
