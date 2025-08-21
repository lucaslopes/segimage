"""
Command-line interface entrypoint.
"""

import click


@click.group()
@click.version_option()
def main():
    """
    segimage - Image segmentation and processing library
    
    Process images using various algorithms and convert between formats.
    """
    pass


# Import command modules to register them with the main group
# The decorators inside these modules reference `main` at import time.
from .commands import process as _process  # noqa: F401,E402
from .commands import inspect as _inspect  # noqa: F401,E402
from .commands import formats as _formats  # noqa: F401,E402
from .commands import info as _info  # noqa: F401,E402


