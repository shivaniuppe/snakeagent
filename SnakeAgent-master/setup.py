from setuptools import setup

setup(
	name='SnakeAgent',
	version='0.1',
	install_requires=[
		'gym==0.21.0',
		'pygame',
		'stable-baselines3[extra]',
		'torch>=1.11',
		'torchvision'
	]
)
