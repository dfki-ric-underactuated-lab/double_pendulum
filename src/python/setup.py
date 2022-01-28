from setuptools import setup, find_packages

setup(
    name='DoublePendulum',
    author='Underactuated Lab DFKI Robotics Innovation Center Bremen',
    version='1.0.0',
    url="https://github.com/dfki-ric-underactuated-lab",
    packages=find_packages(),
    install_requires=[
        # general
        'numpy',
        'matplotlib',
        'bitstring',
        # 'pyyaml',
        'pandas',
        # 'argparse',
        'scipy',
        'sympy',
        'cma',
        # 'mini-cheetah-motor-driver-socketcan',

        # c++ python bindings
        'cython'

        # optimal control
        # 'drake',
        # 'crocoddyl',

        # # reinforcement learning
        # 'tensorflow>=2.6.0',
        # 'pickle5',
        # 'stable_baselines3'
    ],
    classifiers=[
          'Development Status :: 5 - Stable',
          'Environment :: Console',
          'Intended Audience :: Academic Usage',
          'Programming Language :: Python',
          ],
)