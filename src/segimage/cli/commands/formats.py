import click

from segimage.processor import ImageProcessor
from segimage.cli.main import main


@main.command()
def formats():
    """Show supported input and output formats."""
    processor = ImageProcessor()
    formats = processor.get_supported_formats()
    
    click.echo("Supported formats:")
    click.echo(f"  Input:  {', '.join(formats['input'])}")
    click.echo(f"  Output: {', '.join(formats['output'])}")


