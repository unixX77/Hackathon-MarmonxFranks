from setuptools import setup, find_packages
from glob import glob
setup(name='lego_sorter', version='0.1.0', packages=find_packages(),
 data_files=[('share/ament_index/resource_index/packages',['resource/lego_sorter']),
 ('share/lego_sorter',['package.xml']),('share/lego_sorter/config',glob('config/*')),
 ('share/lego_sorter/launch',glob('launch/*'))],
 install_requires=['setuptools','numpy','scipy','PyYAML'], zip_safe=False,
 maintainer='Integration team', maintainer_email='team@example.com', description='FR3 structured LEGO sorter', license='Proprietary',
 entry_points={'console_scripts':['perception = lego_sorter.ros_nodes:perception_main',
 'manipulation = lego_sorter.ros_nodes:manipulation_main','coordinator = lego_sorter.ros_nodes:coordinator_main',
 'lego_preview = lego_sorter.cli:main','sam_worker = lego_sorter.sam_worker:main']})
