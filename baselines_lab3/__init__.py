import os
import sys

# Allow absolute imports even if project is not installed
sys.path.insert(0, os.path.dirname(os.getcwd()))

# Import policies to init policy registry
import baselines_lab3.policies  # noqa # pylint: disable=unused-import
