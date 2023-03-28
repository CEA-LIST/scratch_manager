from setuptools import setup

setup(
    name="scratch_manager",
    entry_points={
        "console_scripts": [
            "scratch_manager=scratch_manager:run_daemon",
        ],
    },
)
