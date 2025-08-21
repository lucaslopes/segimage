import click

from segimage.cli.main import main
from segimage.processors import available_processors


@main.command()
def info():
    """Show information about the segimage library."""
    click.echo("segimage - Image segmentation and processing library")
    click.echo("Version: 0.0.1")
    click.echo("\nAvailable commands:")
    click.echo("  process  - Process an image file")
    click.echo("  inspect  - Inspect MATLAB .mat file contents")
    click.echo("  formats  - Show supported formats")
    click.echo("  info     - Show this information")
    click.echo("\nExample usage:")
    click.echo("  segimage process input.mat output_dir --process-type mat_to_image")
    click.echo("  segimage process input.mat output_dir -t mat_to_image -f png")
    click.echo("  segimage process input.mat output_dir -t mat_to_image -f jpg")
    click.echo("  segimage inspect input.mat")
    click.echo("\nProcessing types:")
    click.echo("  mat_to_image  - Convert MATLAB .mat to image format (PNG, JPG, etc.)")
    click.echo("  inspect       - Analyze MATLAB file structure")
    for name in sorted(available_processors().keys()):
        click.echo(f"  {name:13s} - External processor")
    click.echo("\nOutput formats:")
    click.echo("  png, jpg, jpeg, tif, tiff")


