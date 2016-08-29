from setuptools import setup

setup(
    name='volleyup',
    version=0.1,
    url='http://github.com/jinified/volleyup/',
    author='Jin',
    author_email='jinified@gmail.com',
    description='Beach Volleyball Match Analysis',
    license='MIT',
    keywords='tracking detection volleyball',
    install_requires=['OpenCV>=3.0',
                      ],
    packages=['volleyup'],
)
