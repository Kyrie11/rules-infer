from setuptools import setup, find_packages

setup(
    name='rules-infer',  # 项目名称
    version='0.1',  # 项目版本
    packages=find_packages(),  # 自动查找所有子包
    install_requires=[  # 这里可以添加项目的依赖库
        # 'numpy',
        # 'pandas',
        # 你可以在此添加任何其他依赖库
    ],
    classifiers=[
        "Programming Language :: Python :: 3",  # 支持的 Python 版本
        "License :: OSI Approved :: MIT License",  # 许可证类型
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',  # 需要 Python 版本
)
