from pathlib import Path
import click

from segimage.processor import ImageProcessor
from segimage.cli.main import main
from segimage.utils import write_meta_for_image


@main.command()
@click.argument('input_image_path', type=click.Path(exists=True, path_type=Path))
@click.argument('output_directory', type=click.Path(file_okay=False, path_type=Path))
@click.option('--process-type', '-t', 
              default='mat_to_image',
              help='Type of processing to perform (default: mat_to_image)')
@click.option('--output-format', '-f',
              type=click.Choice(['png', 'jpg', 'jpeg', 'tif', 'tiff']),
              default='png',
              help='Output format (default: png)')
@click.option('--k', '-K', type=click.IntRange(1), default=2, help='Max number of communities/clusters (default: 2)')
@click.option('--palette', type=click.Choice(['bw', 'rainbow']), default='bw', help='Palette for cluster colors (default: bw)')
@click.option('--n-segments', type=int, default=280, help='Approximate number of superpixels for SLICO (default: 280)')
@click.option('--compactness', type=float, default=2.0, help='Compactness for SLIC/SLICO (default: 2.0)')
@click.option('--sigma', type=float, default=1.0, help='Sigma for pre-smoothing in SLIC/SLICO (default: 1.0)')
@click.option('--start-label', type=int, default=1, help='Starting label index for SLIC/SLICO (default: 1)')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--save-meta/--no-save-meta', default=None, help='Write a .meta file with per-pixel details alongside outputs')
@click.pass_obj
def process(ctx, input_image_path: Path, output_directory: Path, process_type: str, 
           output_format: str, k: int, palette: str, n_segments: int, compactness: float, sigma: float, start_label: int, verbose: bool, save_meta: bool | None):
    """
    Process an image file and save the result to the specified output directory.
    
    INPUT_IMAGE_PATH: Path to the input image file
    OUTPUT_DIRECTORY: Directory where the processed image will be saved
    """
    try:
        # Create output directory if it doesn't exist
        output_directory.mkdir(parents=True, exist_ok=True)
        
        # Determine output filename and format
        if output_format:
            # Ensure format has dot prefix
            if not output_format.startswith('.'):
                output_format = '.' + output_format
            output_filename = f"{input_image_path.stem}_processed{output_format}"
        else:
            output_format = '.png'  # Default to PNG
            output_filename = f"{input_image_path.stem}_processed{output_format}"
        
        output_path = output_directory / output_filename
        
        # Determine whether to save .meta: subcommand option overrides global flag
        effective_save_meta = bool(ctx.get('save_meta', False)) if save_meta is None else bool(save_meta)

        if verbose:
            click.echo(f"Processing {input_image_path}")
            click.echo(f"Output will be saved to: {output_path}")
            click.echo(f"Process type: {process_type}")
            click.echo(f"Output format: {output_format}")
            click.echo(f"K (clusters): {k}")
            click.echo(f"Palette: {palette}")
            click.echo(f"SLICO n_segments: {n_segments}")
            click.echo(f"SLICO compactness: {compactness}")
            click.echo(f"SLICO sigma: {sigma}")
            click.echo(f"SLICO start_label: {start_label}")
            click.echo(f"Save .meta: {effective_save_meta}")
        
        # Initialize processor and process image
        processor = ImageProcessor()
        extra_opts = {}
        pt = process_type.lower()
        if pt == 'color_cluster':
            extra_opts = {"K": k, "palette": palette}
        elif pt == 'slico':
            extra_opts = {
                "n_segments": n_segments,
                "compactness": compactness,
                "sigma": sigma,
                "start_label": start_label,
            }
        elif pt == 'lbp':
            extra_opts = {"palette": palette}
        success = processor.process_image(input_image_path, output_path, process_type, **extra_opts)
        
        if success:
            click.echo(f"✅ Successfully processed image to: {output_path}")
            if effective_save_meta:
                try:
                    if write_meta_for_image(output_path):
                        if verbose:
                            click.echo(f"Wrote metadata: {output_path.with_suffix(output_path.suffix + '.meta')}")
                    else:
                        click.echo("Warning: Failed to write .meta file")
                except Exception as e:
                    click.echo(f"Warning: Error writing .meta file: {e}")
        else:
            click.echo("❌ Failed to process image")
            raise click.Abort()
            
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        raise click.Abort()


