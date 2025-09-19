from pathlib import Path
import click

from segimage.processor import ImageProcessor
from segimage.pipelines import get_pipeline, available_pipelines
from segimage.cli.main import main
from segimage.utils import write_meta_for_image
from segimage.graphs import available_graph_builders


@main.command()
@click.argument('input_image_path', type=click.Path(exists=True, path_type=Path))
@click.argument('output_directory', type=click.Path(file_okay=False, path_type=Path))
@click.option('--process-type', '-t', 
              default='mat_to_image',
              help='Type of processing or pipeline to perform (default: mat_to_image). Use names from processors or pipelines (e.g., slico_graph).')
@click.option('--output-format', '-f',
              type=click.Choice(['png', 'jpg', 'jpeg', 'tif', 'tiff', 'npy', 'graphml', 'gml', 'lg', 'lgl', 'edgelist', 'edges', 'txt', 'pickle', 'pkl']),
              default='png',
              help='Output format (default: png)')
@click.option('--k', '-K', '--max-communities', 'k', type=click.IntRange(1), default=2, help='Max number of communities/clusters for hedonic detection (default: 2)')
@click.option('--palette', type=click.Choice(['bw', 'rainbow']), default='bw', help='Palette for cluster colors (default: bw)')
@click.option('--n-segments', type=int, default=280, help='Approximate number of superpixels for SLICO (default: 280)')
@click.option('--compactness', type=float, default=2.0, help='Compactness for SLIC/SLICO (default: 2.0)')
@click.option('--sigma', type=float, default=1.0, help='Sigma for pre-smoothing in SLIC/SLICO (default: 1.0)')
@click.option('--start-label', type=int, default=1, help='Starting label index for SLIC/SLICO (default: 1)')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--save-meta/--no-save-meta', default=None, help='Write a .meta file with per-pixel details alongside outputs')
@click.option('--edge-filter', type=click.Choice(['none', 'lbp_eq', 'lbp', 'gray', 'rgb']), default='none', help='Edge filter for graph processor (default: none)')
@click.option('--edge-similarity', type=click.FloatRange(0.0, 1.0), default=0.0, help='Similarity threshold in [0,1]: 1.0 requires exact match, 0.0 allows any difference (no filtering). Applies to lbp/gray/rgb (ignored for lbp_eq).')
@click.option('--resolution', type=float, default=None, help='Resolution for hedonic detection; if omitted, derived from graph density.')
@click.option('--graph-method', type=str, default=None, help='Graph construction method (default depends on pipeline). Choices: from registry.')
@click.option('--radius', type=int, default=None, help='Graph method param: radius (affinity)')
@click.option('--sigma-i', type=float, default=None, help='Graph method param: sigma_I (affinity)')
@click.option('--sigma-x', type=float, default=None, help='Graph method param: sigma_X (affinity)')
@click.option('--alpha', type=float, default=None, help='Graph method param: alpha (contrast4)')
@click.option('--node-radius', type=int, default=None, help='Graph view param: node radius in pixels (default: 2)')
@click.option('--edge-min', type=click.FloatRange(0.0, 1.0), default=None, help='Graph view param: minimum weight to draw edge in [0,1] (default: 0.0)')
@click.option('--edge-width-max', type=int, default=None, help='Graph view param: maximum edge thickness in pixels (default scales with node size)')
@click.option('--node-mode', type=click.Choice(['pixel', 'superpixel']), default='pixel', help="Node mode for graph builders ('pixel' or 'superpixel').")
@click.option('--enforce-max-communities/--no-enforce-max-communities', default=True, help='If enabled, merge extra communities down to --max-communities after detection (hedonic pipelines).')
@click.pass_obj
def process(ctx, input_image_path: Path, output_directory: Path, process_type: str, 
           output_format: str, k: int, palette: str, n_segments: int, compactness: float, sigma: float, start_label: int, verbose: bool, save_meta: bool | None, edge_filter: str, edge_similarity: float, resolution: float | None, graph_method: str | None, radius: int | None, sigma_i: float | None, sigma_x: float | None, alpha: float | None, node_radius: int | None, edge_min: float | None, edge_width_max: int | None, node_mode: str, enforce_max_communities: bool):
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
            if process_type.lower() == 'graph':
                click.echo(f"Graph edge filter: {edge_filter}")
                click.echo(f"Graph edge similarity: {edge_similarity}")
            if process_type.lower() == 'graph_slico':
                click.echo(f"Graph(SLICO) edge filter: {edge_filter}")
                click.echo(f"Graph(SLICO) edge similarity: {edge_similarity}")
                click.echo(f"Graph(SLICO) n_segments: {n_segments}")
                click.echo(f"Graph(SLICO) compactness: {compactness}")
                click.echo(f"Graph(SLICO) sigma: {sigma}")
                click.echo(f"Graph(SLICO) start_label: {start_label}")
            if process_type.lower() in ('graph_hedonic', 'slico_graph_hedonic'):
                click.echo(f"Hedonic K: {k}")
                click.echo(f"Palette: {palette}")
                click.echo(f"Edge filter: {edge_filter}")
                click.echo(f"Edge similarity: {edge_similarity}")
                click.echo(f"Resolution: {resolution}")
                if graph_method:
                    choices = ', '.join(sorted(available_graph_builders().keys()))
                    click.echo(f"Graph method: {graph_method} (available: {choices})")
                    if radius is not None:
                        click.echo(f"radius: {radius}")
                    if sigma_i is not None:
                        click.echo(f"sigma_I: {sigma_i}")
                    if sigma_x is not None:
                        click.echo(f"sigma_X: {sigma_x}")
                    if alpha is not None:
                        click.echo(f"alpha: {alpha}")
            if process_type.lower() == 'graph_view':
                choices = ', '.join(sorted(available_graph_builders().keys()))
                click.echo(f"Graph view method: {graph_method or 'grid'} (available: {choices})")
                click.echo(f"Node mode: {node_mode}")
                if edge_filter:
                    click.echo(f"Edge filter: {edge_filter}")
                click.echo(f"Edge similarity: {edge_similarity}")
                if radius is not None:
                    click.echo(f"radius: {radius}")
                if sigma_i is not None:
                    click.echo(f"sigma_I: {sigma_i}")
                if sigma_x is not None:
                    click.echo(f"sigma_X: {sigma_x}")
                if alpha is not None:
                    click.echo(f"alpha: {alpha}")
                if node_radius is not None:
                    click.echo(f"node_radius: {node_radius}")
                if edge_min is not None:
                    click.echo(f"edge_min: {edge_min}")
                if edge_width_max is not None:
                    click.echo(f"edge_width_max: {edge_width_max}")
        
        # Initialize processor and process image
        processor = ImageProcessor()
        extra_opts = {}
        pt = process_type.lower()
        # Detect pipeline names
        pipeline = get_pipeline(pt)
        if pipeline is not None:
            # Prepare options for pipelines we ship now
            if pt == 'slico_graph':
                extra_opts = {
                    "n_segments": n_segments,
                    "compactness": compactness,
                    "sigma": sigma,
                    "start_label": start_label,
                    "edge_similarity": edge_similarity,
                }
                if edge_filter != 'none':
                    extra_opts["edge_filter"] = edge_filter
            elif pt == 'graph_hedonic':
                extra_opts = {
                    "K": k,
                    "palette": palette,
                    "edge_similarity": edge_similarity,
                }
                if resolution is not None:
                    extra_opts["resolution"] = float(resolution)
                if edge_filter != 'none':
                    extra_opts["edge_filter"] = edge_filter
                    if graph_method:
                        extra_opts["graph_method"] = graph_method
                    if radius is not None:
                        extra_opts["radius"] = int(radius)
                    if sigma_i is not None:
                        extra_opts["sigma_I"] = float(sigma_i)
                    if sigma_x is not None:
                        extra_opts["sigma_X"] = float(sigma_x)
                if alpha is not None:
                    extra_opts["alpha"] = float(alpha)
            elif pt == 'slico_graph_hedonic':
                extra_opts = {
                    "K": k,
                    "palette": palette,
                    "n_segments": n_segments,
                    "compactness": compactness,
                    "sigma": sigma,
                    "start_label": start_label,
                    "edge_similarity": edge_similarity,
                }
                if resolution is not None:
                    extra_opts["resolution"] = float(resolution)
                if edge_filter != 'none':
                    extra_opts["edge_filter"] = edge_filter
                    if graph_method:
                        extra_opts["graph_method"] = graph_method
                    if radius is not None:
                        extra_opts["radius"] = int(radius)
                    if sigma_i is not None:
                        extra_opts["sigma_I"] = float(sigma_i)
                    if sigma_x is not None:
                        extra_opts["sigma_X"] = float(sigma_x)
                if alpha is not None:
                    extra_opts["alpha"] = float(alpha)
                extra_opts["enforce_max_communities"] = bool(enforce_max_communities)
            elif pt == 'graph_view':
                extra_opts = {
                    "graph_method": graph_method or 'grid',
                    "node_mode": node_mode,
                    "edge_similarity": edge_similarity,
                }
                if edge_filter != 'none':
                    extra_opts["edge_filter"] = edge_filter
                if radius is not None:
                    extra_opts["radius"] = int(radius)
                if sigma_i is not None:
                    extra_opts["sigma_I"] = float(sigma_i)
                if sigma_x is not None:
                    extra_opts["sigma_X"] = float(sigma_x)
                if alpha is not None:
                    extra_opts["alpha"] = float(alpha)
                if node_radius is not None:
                    extra_opts["node_radius"] = int(node_radius)
                if edge_min is not None:
                    extra_opts["edge_min"] = float(edge_min)
                if edge_width_max is not None:
                    extra_opts["edge_width_max"] = int(edge_width_max)
                # Superpixel options
                if node_mode == 'superpixel':
                    extra_opts.update({
                        "n_segments": n_segments,
                        "compactness": compactness,
                        "sigma": sigma,
                        "start_label": start_label,
                    })
            success = bool(pipeline(input_image_path, output_path, **extra_opts))
            if success:
                click.echo(f"✅ Successfully ran pipeline: {process_type}")
            else:
                click.echo("❌ Failed to run pipeline")
                raise click.Abort()
            return
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
        elif pt == 'graph':
            extra_opts = {}
            if edge_filter != 'none':
                extra_opts["edge_filter"] = edge_filter
            # similarity is harmless to pass; processor will validate range and ignore when irrelevant
            extra_opts["edge_similarity"] = edge_similarity
        elif pt == 'graph_slico':
            extra_opts = {
                "n_segments": n_segments,
                "compactness": compactness,
                "sigma": sigma,
                "start_label": start_label,
                "edge_similarity": edge_similarity,
            }
            if edge_filter != 'none':
                extra_opts["edge_filter"] = edge_filter
        success = processor.process_image(input_image_path, output_path, process_type, **extra_opts)
        
        if success:
            click.echo(f"✅ Successfully processed image to: {output_path}")
            # Only write .meta for image outputs
            image_suffixes = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
            if effective_save_meta and output_path.suffix.lower() in image_suffixes:
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


