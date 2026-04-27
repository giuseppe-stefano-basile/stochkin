"""Build configuration for optional compiled extensions."""

import sys

import numpy as np
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class OptionalBuildExt(build_ext):
    """Build extensions when possible, but keep pure-Python installs working."""

    def run(self):
        try:
            super().run()
        except Exception as exc:
            self.warn(f"optional compiled extensions were not built: {exc}")

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception as exc:
            self.warn(f"optional extension {ext.name!r} was not built: {exc}")


if sys.platform == "win32":
    extra_compile_args = ["/O2"]
else:
    extra_compile_args = ["-std=c++11", "-O3"]


setup(
    cmdclass={"build_ext": OptionalBuildExt},
    ext_modules=[
        Extension(
            "stochkin._fast_langevin1d",
            ["stochkin/_fast_langevin1d.cpp"],
            include_dirs=[np.get_include()],
            language="c++",
            extra_compile_args=extra_compile_args,
        )
    ]
)
