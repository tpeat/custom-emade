from setuptools import setup


setup(name = 'GPFramework',
      version = '1.0',
      author='Jason Zutty',
      author_email='jasonzutty@gmail.com',
      packages = ['GPFramework'],
      package_dir = {'GPFramework': 'src/GPFramework'},
      install_requires = ['pillow', 'psutil', 'lxml', 'hmmlearn', 'scikit-learn', 'deap',
	'scoop', 'pandas', 'scipy', 'numpy', 'matplotlib', 'multiprocess',
    'sqlalchemy', 'PyWavelets', 'keras', 'mysqlclient', 'networkx', 'lmfit', 
    'scikit-image', 'opencv-python', 'pymysql'],
      )
