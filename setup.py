import setuptools

with open('requirements.txt') as f:
    reqs = [line for line in f]
reqs.append('pytest-spec')

with open('VERSION') as f:
    VERSION = f.read()

setuptools.setup(
    name="slaid",  # Replace with your own username
    version=VERSION,
    description="AI for automatic analys of slides",
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.6',
    install_requires=reqs,
    scripts=['bin/classify.py', 'bin/render_mask.py'],
    package_data={'': ['resources/models/*']},
    include_package_data=True)
