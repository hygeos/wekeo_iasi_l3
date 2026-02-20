import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
import xarray as xr
from pathlib import Path

# ===== CONFIGURATION PARAMETERS =====
BACKGROUND_COLOR = 'black'
FOREGROUND_COLOR = 'white'
GRIDLINES_COLOR = 'gray'
USE_LOG_SCALE = False
FONT_SIZE = 14
# ====================================


def plot_L3_IASI(
    dataset: xr.Dataset,
    variable: str,
    title: str = None,
    use_log_scale: bool = USE_LOG_SCALE,
    figsize: tuple = (14, 7),
    cmap: str = 'viridis',
    vmin: float = None,
    vmax: float = None,
    save_fig_dir: str = None,
):
    """
    Plot gridded L3 IASI data from an xarray Dataset.
    
    Parameters:
    -----------
    dataset : xr.Dataset
        Gridded IASI dataset with latitude and longitude coordinates
    variable : str
        Variable to plot (e.g., 'INTEGRATED_CO_MEAN', 'INTEGRATED_CO_CNT')
    title : str, optional
        Custom title for the plot. If None, auto-generated based on variable name
    use_log_scale : bool, optional
        Whether to use logarithmic scale for the colorbar (default: False)
    figsize : tuple, optional
        Figure size (width, height) in inches
    cmap : str, optional
        Colormap name (default: 'viridis')
    vmin : float, optional
        Minimum value for the colormap scale. If None, auto-determined from data
    vmax : float, optional
        Maximum value for the colormap scale. If None, auto-determined from data
    save_fig_dir : str, optional
        Directory path to save the figure. If None, figure is not saved.
        Filename will be auto-generated based on variable name.
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    # Validate variable exists in dataset
    if variable not in dataset.data_vars:
        raise ValueError(
            f"Variable '{variable}' not found in dataset. "
            f"Available variables: {list(dataset.data_vars)}"
        )
    
    # Extract data
    data = dataset[variable]
    lats = dataset['latitude'].values
    lons = dataset['longitude'].values
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize, facecolor=BACKGROUND_COLOR)
    ax.set_facecolor(BACKGROUND_COLOR)
    
    # Add gridlines
    ax.grid(True, linewidth=0.5, color=GRIDLINES_COLOR, alpha=0.3, linestyle='--')
    ax.set_xlabel('Longitude', fontsize=FONT_SIZE, color=FOREGROUND_COLOR)
    ax.set_ylabel('Latitude', fontsize=FONT_SIZE, color=FOREGROUND_COLOR)
    ax.tick_params(colors=FOREGROUND_COLOR, labelsize=10)
    
    # Determine if we should use log scale based on variable type and data range
    is_count_variable = 'cnt' in variable.lower() or 'count' in variable.lower()
    
    # Prepare data for plotting
    plot_data = data.values.copy()
    
    # For count variables, set zero values to NaN so they appear transparent
    if is_count_variable:
        plot_data = plot_data.astype(float)
        plot_data[plot_data <= 0] = np.nan
    
    # Handle NaN values - they will be transparent in the plot
    valid_data = plot_data[~np.isnan(plot_data)]
    
    if len(valid_data) == 0:
        print(f"Warning: No valid data to plot for variable '{variable}'")
        return fig, ax
    
    # Determine normalization
    if use_log_scale and not is_count_variable:
        # For log scale, use only positive values
        positive_data = valid_data[valid_data > 0]
        if len(positive_data) > 0:
            # Use provided vmin/vmax or auto-determine from data
            if vmin is None:
                vmin = max(positive_data.min(), 0.01)
            if vmax is None:
                vmax = positive_data.max()
            norm = LogNorm(vmin=vmin, vmax=vmax)
        else:
            print(f"Warning: No positive values for log scale. Using linear scale.")
            norm = None
    else:
        norm = None
    
    # Plot the gridded data
    if norm is not None:
        im = ax.pcolormesh(
            lons, lats, plot_data,
            cmap=cmap,
            norm=norm,
            shading='auto'
        )
    else:
        # Linear scale - apply vmin/vmax directly if provided
        im = ax.pcolormesh(
            lons, lats, plot_data,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading='auto'
        )
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
    
    # Determine colorbar label
    if title is None:
        # Auto-generate title from variable name
        title = variable.replace('_', ' ').title()
    
    # Get unit from dataset attributes if available
    unit = data.attrs.get('unit', '')
    long_name = data.attrs.get('long_name', variable)
    
    # Set colorbar label
    if 'mean' in variable.lower():
        unit_str = f' ({unit})' if unit else ''
        scale_note = ' [log scale]' if (use_log_scale and not is_count_variable) else ''
        cbar.set_label(f'{long_name}{unit_str}{scale_note}', fontsize=FONT_SIZE, color=FOREGROUND_COLOR)
    elif 'cnt' in variable.lower() or 'count' in variable.lower():
        cbar.set_label(f'{long_name} (number of observations)', fontsize=FONT_SIZE, color=FOREGROUND_COLOR)
    else:
        unit_str = f' ({unit})' if unit else ''
        cbar.set_label(f'{long_name}{unit_str}', fontsize=FONT_SIZE, color=FOREGROUND_COLOR)
    
    cbar.ax.tick_params(colors=FOREGROUND_COLOR)
    
    # Add statistics to title
    n_valid_pixels = np.sum(~np.isnan(plot_data))
    n_nonzero_pixels = np.sum((~np.isnan(plot_data)) & (plot_data > 0))
    
    title_with_stats = f'{title}\n({n_nonzero_pixels} non-zero pixels, {n_valid_pixels} valid pixels)'
    ax.set_title(title_with_stats, fontsize=FONT_SIZE * 1.25, fontweight='bold', color=FOREGROUND_COLOR, pad=10)
    
    # Set axis limits
    ax.set_xlim(lons.min(), lons.max())
    ax.set_ylim(lats.min(), lats.max())
    
    plt.tight_layout()
    
    # Save figure if directory is provided
    if save_fig_dir is not None:
        save_dir = Path(save_fig_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        # Try to get date from dataset attributes
        day = dataset.attrs.get('date', 'unknown')
        if day == 'unknown' and 'description' in dataset.attrs:
            # Try to extract date from description
            desc = dataset.attrs['description']
            import re
            match = re.search(r'(\d{4}-\d{2}-\d{2}|\d{8})', desc)
            if match:
                day = match.group(1)
        filename = f"IASI_L3_{day}_{variable}.png"
        filepath = save_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor=BACKGROUND_COLOR)
        print(f"Figure saved to: {filepath}")
    
    return fig, ax
