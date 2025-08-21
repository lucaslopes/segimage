from pathlib import Path
import click

from segimage.processor import ImageProcessor
from segimage.cli.main import main


@main.command()
@click.argument('input_image_path', type=click.Path(exists=True, path_type=Path))
def inspect(input_image_path: Path):
    """
    Inspect the contents of a MATLAB .mat file to understand its structure.
    
    INPUT_IMAGE_PATH: Path to the input MATLAB .mat file
    """
    try:
        processor = ImageProcessor()
        success = processor.inspect_mat_file(input_image_path)
        
        if not success:
            click.echo("❌ Failed to inspect file")
            raise click.Abort()
            
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        raise click.Abort()


