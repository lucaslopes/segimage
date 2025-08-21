"""
Command-line interface entrypoint.
"""

import click


@click.group()
@click.version_option()
@click.option(
    "--save-meta/--no-save-meta",
    default=False,
    help="If enabled, write a .meta file with per-pixel details alongside outputs.",
)
@click.pass_context
def main(ctx, save_meta: bool):
    """
    segimage - Image segmentation and processing library
    
    Process images using various algorithms and convert between formats.
    """
    ctx.ensure_object(dict)
    ctx.obj["save_meta"] = bool(save_meta)


# Import command modules to register them with the main group
# The decorators inside these modules reference `main` at import time.
from .commands import process as _process  # noqa: F401,E402
from .commands import inspect as _inspect  # noqa: F401,E402
from .commands import formats as _formats  # noqa: F401,E402
from .commands import info as _info  # noqa: F401,E402


