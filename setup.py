import re
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


with open("basedet/__init__.py", "r") as f:
    version = re.search(
        r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
        f.read(), re.MULTILINE
    ).group(1)


with open("requirements.txt", "r") as f:
    reqs = [x.strip() for x in f.readlines()]


setuptools.setup(
    name="basedet",
    version=version,
    author="basedet team",
    author_email="wangfeng02@megvii.com",
    description="megvii basedet team codebase",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=None,
    packages=setuptools.find_packages(),
    classifiers=["Programming Language :: Python :: 3", "Operating System :: OS Independent"],
    install_requires=reqs,
    entry_points={
        "console_scripts": [
            "basedet_train=basedet.tools.det_train:main",
            "basedet_test=basedet.tools.det_test:main",
            "basedet_profile=basedet.tools.profile_net:main",
            "basedet_dump_cfg=basedet.tools.dump_cfg:main",
            "basedet_trace=basedet.tools.trace_net:main",
        ]
    },
)
