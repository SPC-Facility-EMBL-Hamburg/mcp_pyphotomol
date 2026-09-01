
import glob
import json
import os
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any

from mcp.server.mcpserver.exceptions import ToolError

import numpy as np
import pandas as pd

from pyphotomol import plot_calibration as pm_plot_calibration
from pyphotomol import plot_histogram as pm_plot_histogram
from pyphotomol import plot_histograms_and_fits as pm_plot_histograms_and_fits

from ..config import DATA_DIR, EXAMPLE_DATA_DIR
from ..mcp import mcp
from ..server import (
    AXIS_CONFIG,
    LAYOUT_CONFIG,
    LEGEND_CONFIG,
    MP_ANALYZER,
    MP_CALIBRATOR,
    PLOT_CONFIG,
)

def append_function_call_to_logbook(function_name: str, params: dict) -> None:
    """
    Append a function call to the MCP logbook.
    
    Parameters
    ----------
    function_name : str
        The name of the function being called.
    params : dict
        The parameters passed to the function.
    """
    logbook_path = os.path.join(DATA_DIR, 'mcp_logbook.txt')
    with open(logbook_path, 'a') as f:
        f.write(f"{datetime.now().isoformat()} - {function_name} called with params: {params}\n")

def tool_with_log():
    """
    Decorator that combines @mcp.tool() with automatic logging to the MCP logbook.

    Returns
    -------
    Callable
        Decorator that registers a function as an MCP tool and logs calls.
    """
    def decorator(func):
        """
        Wrap a function with call logging before MCP tool registration.

        Parameters
        ----------
        func : Callable
            Function to register as an MCP tool.
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            """
            Log a tool call and execute the wrapped function.

            Parameters
            ----------
            *args : Any
                Positional arguments for the wrapped function.
            **kwargs : Any
                Keyword arguments for the wrapped function.
            """
            # Get function signature to map args to parameter names
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Convert all parameters to JSON-serializable format
            params = {}
            for name, value in bound_args.arguments.items():
                if isinstance(value, Path):
                    params[name] = str(value)
                else:
                    params[name] = value
            
            # Log the function call
            append_function_call_to_logbook(func.__name__, params)
            
            # Execute the original function
            try:
                return func(*args, **kwargs)
            except ValueError as exc:
                raise ToolError(str(exc)) from exc
        
        # Apply the @mcp.tool() decorator to the wrapper
        return mcp.tool()(wrapper)
    
    return decorator

@tool_with_log()
def get_user_name() -> str:
    """
    Get the current user name
    Useful to finding the path to a folder when the user wants to import a file or a folder
    Returns
    -------
    str
        The current user name.
    """

    import getpass
    return getpass.getuser()

@tool_with_log()
def list_MP_files_in_folder(folder_path: Path) -> str:
    """
    List all files in a folder.

    Parameters
    ----------
    folder_path : Path
        The path to the folder to list files from.

    Returns
    -------
    str
        A comma-separated list of file names in the folder.
    
    Notes
    -----
    This function only lists files with the ".h5" or ".csv" extension.
    """
    assert isinstance(folder_path, Path)

    # List files
    return ", ".join(f.name for f in folder_path.iterdir() if f.is_file() and f.suffix in [".h5", ".csv"])

@tool_with_log()
def reset_analyzer() -> str:
    """
    Reset the MP_ANALYZER instance.
    
    Returns
    -------
    str
        A message indicating that the analyzer has been reset.
    """
    MP_ANALYZER.models = {}
    MP_ANALYZER.batch_logbook = []
    
    return "MP_ANALYZER instance has been reset."

@tool_with_log()
def reset_calibrator() -> str:
    """
    Reset the MP_CALIBRATOR instance.

    Returns
    -------
    str
        A message indicating that the calibrator has been reset.
    """
    MP_CALIBRATOR.models = {}
    MP_CALIBRATOR.batch_logbook = []

    return "MP_CALIBRATOR instance has been reset."

@tool_with_log()
def get_model_names(calibrator: bool = False) -> list[str]:
    """
    Get the names of the models in the MP_ANALYZER or MP_CALIBRATOR instance.

    Parameters
    ----------
    calibrator : bool
        If True, get the model names from the MP_CALIBRATOR instance.
        If False, get the model names from the MP_ANALYZER instance.

    Returns
    -------
    list[str]
        A list of model names.
    """
    py_object = MP_CALIBRATOR if calibrator else MP_ANALYZER
    return list(py_object.models.keys())

@tool_with_log()
def import_single_file(file_path: Path,name: str = '',calibrator: bool = False) -> str:
    """
    Import data files into the MP_ANALYZER instance.
    
    Parameters
    ----------
    file_path : Path
        The path to the data file to be imported.
    name : str
        The name to assign to the imported data. 
        If the name is not provided, it will be derived from the file name.
    calibrator : bool
        If True, the data will be imported into the MP_CALIBRATOR instance instead of MP_ANALYZER.
        This is useful for calibration data.
    Returns
    -------
    str
        A message indicating the success or failure of the import operation.
    """

    assert isinstance(file_path, Path)  

    py_object = MP_CALIBRATOR if calibrator else MP_ANALYZER

    n_models = len(py_object.models)

    # If no name is provided, use the file basename without extension
    if not name:
        name = str(file_path.stem)

    py_object.import_files([str(file_path)], names=[name])

    # Verify that we have one more model
    if len(py_object.models) == n_models + 1:
        return f"Data imported successfully from {file_path}."
    else:
        return f"Data import failed for {file_path}."

@tool_with_log()
def load_example_data() -> str:
    """
    Load example data into the MP_ANALYZER instance.
    The example data contains simulated MP measurements of a protein at different concentrations
    The proteins can dimerize: A + A <=> A2
    Returns
    -------
    str
        A message indicating that the example data has been loaded successfully.
    """

    # Reset the MP_ANALYZER instance
    MP_ANALYZER.models = {}
    MP_ANALYZER.batch_logbook = []

    # Sort them by concentration 
    # Get all monomer files with concentrations in filename, sorted by concentration
    files = sorted(glob.glob(f"{EXAMPLE_DATA_DIR}/masses_monomer_*nM.csv"),
                   key=lambda x: float(x.split('_')[-1].replace('nM.csv', '')))

    # Extract the total monomer concentration from each filename
    concentrations = [float(f.split('_')[-1].replace('nM.csv', '')) for f in files]
    # Convert to string and include 'nM' suffix
    concentration_strings = [f"{c} nM" for c in concentrations]

    MP_ANALYZER.import_files(files,names=concentration_strings)

    return "Example data loaded successfully."

@tool_with_log()
def load_example_data_for_calibration() -> str:

    """
    Load example data (contrasts) in the MP_CALIBRATOR instance.
    As a result, the MP_CALIBRATOR will have two MP files. 
    One file with one peak, and another file with two peaks.

    We recommend running the tool 'calibrate' afterwards with the following parameters:
        known_standards=[[148,66],[480]] 

    Returns
    -------
    str
        A message indicating that the example data has been loaded successfully.
    """

    MP_CALIBRATOR.models = {}
    MP_CALIBRATOR.batch_logbook = []

    file = os.path.join(EXAMPLE_DATA_DIR, 'contrasts.csv')

    files = [file, file]
    names = ['file1', 'file2']

    MP_CALIBRATOR.import_files(files, names=names)

    # Artificially remove contrasts so we simulate two different files
    # Only required here - do not do this in real analysis
    MP_CALIBRATOR.models['file2'].contrasts = MP_CALIBRATOR.models['file2'].contrasts[MP_CALIBRATOR.models['file2'].contrasts < -0.02]
    MP_CALIBRATOR.models['file1'].contrasts = MP_CALIBRATOR.models['file1'].contrasts[MP_CALIBRATOR.models['file1'].contrasts > -0.02]

    # Create the histogram - same window and bin width for all files
    MP_CALIBRATOR.apply_to_all('create_histogram', use_masses=False, window=[-0.05, 0], bin_width=0.0004)

    # Set the PLOT_CONFIG for the histograms
    PLOT_CONFIG.contrasts = True

    return "Example calibration data loaded successfully."

@tool_with_log()
def import_folder(folder_path: Path, pattern: str = '', calibrator: bool = False) -> str:
    """
    Import all data files from a folder into the MP_ANALYZER instance.
    To create the histograms, proceed with the `create_histogram_manual` or `create_histogram_automatic` tool.

    Parameters
    ----------
    folder_path : Path
        The path to the folder containing data files to be imported.
    pattern : str
        A pattern to filter the files to be imported.
        If not provided, all files in the folder will be imported.
        For example, if the pattern is 'monomer', only files containing 'monomer' in their name will be imported.
    calibrator : bool
        If True, the data will be imported into the MP_CALIBRATOR instance instead of MP_ANALYZER.
        This is useful for calibration data.
    Returns
    -------
    str
        A message indicating the success or failure of the import operation.
    Note
    ----
    After importing the data, the natural step is to call
    the tool 'create_histogram_manual' or 'create_histogram_automatic'.
    """

    assert isinstance(folder_path, Path)

    py_object = MP_CALIBRATOR if calibrator else MP_ANALYZER

    n_models = len(py_object.models)

    # Get all files in the folder
    files = list(folder_path.glob('*'))
    # Filter out non .csv or .h5 files
    files = [f for f in files if f.suffix in ['.csv', '.h5']]

    # Filter files by pattern, for example with the word 'monomer'
    if pattern:
        files = [f for f in files if pattern in f.name]

    if not files:
        return f"No files found in {folder_path}."

    py_object.import_files([str(f) for f in files])

    # Verify that we have added as many models as files
    if len(py_object.models) == n_models + len(files):
        return f"{len(files)} files were imported successfully from {folder_path}."
    else:
        return f"Data import failed for {folder_path}."

# This function an internal helper and therefore is not decorated with @tool_with_log() because it is not meant to be called directly by the user     
def create_histogram(min_value: float | None = None, 
                     max_value: float | None = None, 
                     bin_width: float | None = None, 
                     use_masses: bool = True,
                     calibrator: bool = False) -> str:
    """
    Create the histograms with the specified parameters.
    To let the function infer the min_value, max_value and bin_width, leave them as None.

    Parameters
    ----------
    min_value : float
        The minimum value for the histogram.
        If None, it will be inferred from the data.
    max_value : float
        The maximum value for the histogram.
        If None, it will be inferred from the data.
    bin_width : float
        The size of each bin in the histogram.
        If None, it will be inferred from the data.
    use_masses : bool
        If True, the histogram will be created using mass values.
        If False, it will use contrast values.
    calibrator : bool
        If True, the histogram will be created for the MP_CALIBRATOR instance instead of MP_ANALYZER.
        This is useful for calibration data.
    Returns
    -------
    str
        A message indicating the success or failure of the histogram creation,
        and the parameters used for the histogram.
    """

    assert isinstance(min_value, (int, float, type(None)))
    assert isinstance(max_value, (int, float, type(None)))
    assert isinstance(bin_width, (int, float, type(None)))

    py_object = MP_CALIBRATOR if calibrator else MP_ANALYZER

    # Verify if we need to find the bin_size, min_value, max_value
    find_hist_param = any(param is None for param in [bin_width, min_value, max_value])

    values = []

    # Verify that we have the data and raise and error if not
    model_names = list(py_object.models.keys())
    for model_name in model_names:
        model = py_object.models[model_name]

        # Raise an error if we want to use masses and do not have them
        if use_masses:
            if model.masses is None:
                raise ValueError(f"Mass data is missing for model {model_name}.")
            if find_hist_param:
                values.append(model.masses)

        # Raise an error if we want to use contrasts and we do not have them
        if not use_masses:
            if model.contrasts is None:
                raise ValueError(f"Contrast data is missing for model {model_name}.")
            if find_hist_param:
                values.append(model.contrasts)

    if find_hist_param:
        # Concatenate all values into a single array
        all_values = np.concatenate(values) 

        # Create an histogram with 1000 bins
        counts, bin_edges = np.histogram(all_values, bins=1000)

        # Find the maximum number of counts
        max_counts = np.max(counts)

        mask = counts >= 0.001 * max_counts
        
        min_value_h = bin_edges[np.argmax(mask)]
        max_value_h = bin_edges[np.where(mask)[0][-1] + 1]

        # Set the lower bound for masses to 0 if the min value is not set
        # Set the upper bound for contrasts to 0 if the max value is not set
        if use_masses:
            if min_value is None:
                min_value = 0
            if max_value is None:
                max_value = max_value_h
        else:
            if max_value is None:
                max_value = 0
            if min_value is None:
                min_value = min_value_h

        if bin_width is None:
            if use_masses:
                if max_value_h < 500:
                    bin_width = 8
                elif max_value_h < 1000:
                    bin_width = 10
                else:
                    bin_width = 12
            else:
                bin_width = 0.0004

    window = [min_value, max_value]

    py_object.apply_to_all(
        'create_histogram',
        window=window,
        bin_width=bin_width,
        use_masses=use_masses
    )

    # Return a message containing the bin width, the min and max values and if masses or contrasts were used
    return (f"Histograms were created successfully with the following parameters:\n"
            f"Bin width: {bin_width}\n"
            f"Min value: {min_value}\n"
            f"Max value: {max_value}\n"
            f"Using masses: {use_masses}\n"
            f"Using contrasts: {not use_masses}")

@tool_with_log()
def create_histogram_manual(
    min_value: float,
    max_value: float,
    bin_width: float,
    use_masses: bool = True,
    calibrator: bool = False) -> str:
    """
    Create the histograms using parameters given by the user

    Parameters
    ----------
    min_value : float
        The minimum value for the histogram.
    max_value : float
        The maximum value for the histogram.
    bin_width : float
        The width of the bins for the histogram.
    use_masses : bool
        If True, the histogram will be created using mass values.
    calibrator : bool
        If True, the histogram will be created for the MP_CALIBRATOR instance
        If False, the histogram will be created for the MP_ANALYZER instance.

    Returns
    -------
    str
        A message indicating the success or failure of the histogram creation,
        and the parameters used for the histogram.
    
    Notes
    -----
    The following default values could work.

    For contrast data, try::

        min_value=-0.1, max_value=0, bin_width=0.0004, use_masses=False

    For mass data::

        min_value=0, max_value=1000, bin_width=10, use_masses=True
    """

    return create_histogram(
        min_value=min_value,
        max_value=max_value,
        bin_width=bin_width,
        use_masses=use_masses,
        calibrator=calibrator
    )

@tool_with_log()
def create_histogram_automatic(
    use_masses: bool = True,
    calibrator: bool = False) -> str:
    """
    Let pyphotomol find suitable histogram parameters automatically.
    And then create the histograms.

    Parameters
    ----------
    use_masses : bool
        If True, the histogram will be created using mass values.
    calibrator : bool
        If True, the histogram will be created for the MP_CALIBRATOR instance
        If False, the histogram will be created for the MP_ANALYZER instance.

    Returns
    -------
    str
        A message indicating the success or failure of the histogram creation,
        and the parameters used for the histogram.
    """
    
    return create_histogram(
        min_value=None,
        max_value=None,
        bin_width=None,
        use_masses=use_masses,
        calibrator=calibrator
    )

@tool_with_log()
def update_plot_config(
    plot_width: int = 1000,
    plot_height: int = 600, 
    plot_type: str = 'png', 
    font_size: int = 14, 
    normalize: bool = False, 
    contrasts: bool = False, 
    cst_factor_for_contrast: float = 1, 
    x_range: list[float] | None = None) -> str:

    """
    Edit the plot configuration for the histograms.

    Parameters
    ----------
    plot_width : int
        The width of the plot in pixels.
    plot_height : int
        The height of the plot in pixels.
    plot_type : str
        The type of the plot, e.g., 'png', 'svg', or 'jpg'.
    font_size : int
        The font size for the plot
    normalize : bool
        If True, the histogram will be normalized.
    contrasts : bool
        If True, the histogram will be created using contrast values.
    cst_factor_for_contrast : float
        The contrast factor to be used when creating the histogram with contrast values.
    x_range : list[float] | None
        The x-axis range for the histogram. 
        For example, ``[0, 400]``.
        If None, the default range will be used.

    Returns
    -------
    str
        A message indicating the success or failure of the plot configuration update.
    """

    assert isinstance(plot_width, int)
    assert isinstance(plot_height, int)
    assert isinstance(plot_type, str)
    assert isinstance(font_size, int)
    assert isinstance(normalize, bool)
    assert isinstance(contrasts, bool)
    assert isinstance(cst_factor_for_contrast, (int, float))

    if x_range is not None:
        assert isinstance(x_range, list) and len(x_range) == 2
        assert all(isinstance(x, (int, float)) for x in x_range)

    PLOT_CONFIG.plot_width = plot_width
    PLOT_CONFIG.plot_height = plot_height
    PLOT_CONFIG.plot_type = plot_type
    PLOT_CONFIG.font_size = font_size
    PLOT_CONFIG.normalize = normalize
    PLOT_CONFIG.contrasts = contrasts
    PLOT_CONFIG.cst_factor_for_contrast = cst_factor_for_contrast
    PLOT_CONFIG.x_range = [x_range[0], x_range[1]] if x_range else None

    return "Plot configuration updated successfully."

@tool_with_log()
def update_legend_config(
    add_masses_to_legend: bool = True, 
    add_percentage_to_legend: bool = False, 
    add_labels: bool = True, 
    add_percentages: bool = True, 
    line_width: int = 3) -> str:
    """
    Edit the legend configuration for the histograms.

    Parameters
    ----------
    add_masses_to_legend : bool
        If True, masses (or contrasts) will be added to the legend.
        The masses/contrasts are obtained from a multi-gaussian fit.
    add_percentage_to_legend : bool
        If True, the percentage of counts will be added to the legend.
    add_labels : bool
        If True, labels will be added to the plot.
    add_percentages : bool
        If True, percentages will be added to the plot.
    line_width : int
        The width of the lines in the legend.

    Returns 
    -------
    str
        A message indicating the success or failure of the legend configuration update.
    """

    assert isinstance(add_masses_to_legend, bool)
    assert isinstance(add_percentage_to_legend, bool)
    assert isinstance(add_labels, bool)
    assert isinstance(add_percentages, bool)        
    assert isinstance(line_width, int)

    LEGEND_CONFIG.add_masses_to_legend = add_masses_to_legend
    LEGEND_CONFIG.add_percentage_to_legend = add_percentage_to_legend
    LEGEND_CONFIG.add_labels = add_labels
    LEGEND_CONFIG.add_percentages = add_percentages
    LEGEND_CONFIG.line_width = line_width   

    return "Legend configuration updated successfully."

@tool_with_log()
def update_layout_config(
    stacked: bool = True, 
    show_subplot_titles: bool = False, 
    vertical_spacing: float = 0.04, 
    shared_yaxes: bool = True, 
    extra_padding_y_label: float = 0.04) -> str:
    """
    Edit the layout configuration for the histograms.

    Parameters
    ----------
    stacked : bool
        If True, the histograms will be stacked. There will be as many rows as input files.
    show_subplot_titles : bool
        If True, the subplot titles will be shown.
    vertical_spacing : float
        The vertical spacing between subplots.
    shared_yaxes : bool
        If True, the y-axes will be shared across subplots.
    extra_padding_y_label : float
        Extra padding for the y-axis label. Important for the y-axis label to be visible.
        Only used if the plots are stacked.

    Returns
    -------
    str
        A message indicating the success or failure of the layout configuration update.
    """
    assert isinstance(stacked, bool)
    assert isinstance(show_subplot_titles, bool)
    assert isinstance(vertical_spacing, (int, float))
    assert isinstance(shared_yaxes, bool)
    assert isinstance(extra_padding_y_label, (int, float))

    LAYOUT_CONFIG.stacked = stacked
    LAYOUT_CONFIG.show_subplot_titles = show_subplot_titles
    LAYOUT_CONFIG.vertical_spacing = vertical_spacing
    LAYOUT_CONFIG.shared_yaxes = shared_yaxes
    LAYOUT_CONFIG.extra_padding_y_label = extra_padding_y_label

    return "Layout configuration updated successfully." 

@tool_with_log()
def update_axis_config(
    showgrid_x: bool = True, 
    showgrid_y: bool = True, 
    n_y_axis_ticks: int = 3, 
    axis_linewidth: int = 1, 
    axis_tickwidth: int = 1, 
    axis_gridwidth: int = 1) -> str:
    """
    Edit the axis configuration for the histograms.
    Parameters
    ----------
    showgrid_x : bool
        If True, the x-axis grid will be shown.
    showgrid_y : bool
        If True, the y-axis grid will be shown.
    n_y_axis_ticks : int
        The number of ticks on the y-axis.
    axis_linewidth : int
        The width of the axis lines.
    axis_tickwidth : int
        The width of the axis ticks.
    axis_gridwidth : int
        The width of the axis grid lines.
    Returns
    -------
    str
        A message indicating the success or failure of the axis configuration update.
    """

    assert isinstance(showgrid_x, bool)
    assert isinstance(showgrid_y, bool)
    assert isinstance(n_y_axis_ticks, int)
    assert isinstance(axis_linewidth, int)
    assert isinstance(axis_tickwidth, int)
    assert isinstance(axis_gridwidth, int)

    AXIS_CONFIG.showgrid_x = showgrid_x
    AXIS_CONFIG.showgrid_y = showgrid_y
    AXIS_CONFIG.n_y_axis_ticks = n_y_axis_ticks
    AXIS_CONFIG.axis_linewidth = axis_linewidth     
    AXIS_CONFIG.axis_tickwidth = axis_tickwidth
    AXIS_CONFIG.axis_gridwidth = axis_gridwidth

    return "Axis configuration updated successfully."

@tool_with_log()
def plot_histograms(colors_hist:  list[str] | str | None = None ,
                    save_as_html: bool = False,
                    calibrator: bool = False) -> str:

    """    
    Plot the histograms using the current configuration.

    Parameters
    ----------
    colors_hist : list[str] | str | None
        A list of colors to be used for the histograms.
        One color per file 
        Each color should be a valid hex color code (e.g., '#FF5733') or a named color (e.g., 'red').
        If None, default colors will be used.
        If a single string is provided, it will be used for all histograms
    save_as_html : bool
        If True, the plot will be saved as an HTML file.
    calibrator : bool
        If True, the histograms will be plotted for the MP_CALIBRATOR instance instead than MP_ANALYZER.
        This is useful for calibration data.
    Returns
    -------
    str
        The path to the saved plot
    Note
    ----
    This function will only work if the tool 'create_histogram_manual' or
    the tool 'create_histogram_automatic' has been called first.
    """

    py_object = MP_CALIBRATOR if calibrator else MP_ANALYZER

    # If colors_hist is a single string, convert it to a list with one element
    if isinstance(colors_hist, str):
        colors_hist = [colors_hist] * len(py_object.models)

    time_str = datetime.now().strftime('%H-%M-%S')
    if save_as_html:

        fig = pm_plot_histogram(
            py_object, 
            colors_hist=colors_hist, 
            plot_config=PLOT_CONFIG,
            layout_config=LAYOUT_CONFIG,
            axis_config=AXIS_CONFIG)

        html_path = os.path.join(DATA_DIR, f"histogram_{time_str}.html")
        fig.write_html(html_path)
    
    # For a strange reason the padding of the y-axis label in HTML figures is larger 
    # than in static figures. So we re do the plot with larger spacing

    LAYOUT_CONFIG.extra_padding_y_label = LAYOUT_CONFIG.extra_padding_y_label + 0.02

    fig = pm_plot_histogram(
        py_object, 
        colors_hist=colors_hist, 
        plot_config=PLOT_CONFIG,
        layout_config=LAYOUT_CONFIG,
        axis_config=AXIS_CONFIG)
    
    # Reset back AXIS_CONFIG
    LAYOUT_CONFIG.extra_padding_y_label = LAYOUT_CONFIG.extra_padding_y_label - 0.02

    # Export it also with the PLOT_CONFIG.plot_type
    plot_type = PLOT_CONFIG.plot_type.lower()
    if plot_type in ['png', 'jpg', 'jpeg','svg']:
        image_path = os.path.join(DATA_DIR, f"histogram_{time_str}.{plot_type}")
        fig.write_image(image_path,
                        width=PLOT_CONFIG.plot_width, 
                        height=PLOT_CONFIG.plot_height)
        return f"Histogram plot saved as {plot_type} at {image_path}."
    else:
        return "Histogram plot created successfully, but no valid image format was specified for saving."

@tool_with_log()
def guess_peaks(
    min_height: float = 10,
    min_distance: float = 4,
    prominence: float = 4,
    calibrator: bool = False,
    experiment: str = 'all'
) -> str:
    """
    Automatically detect histogram peaks.

    Parameters
    ----------
    min_height : float
        The minimum height of the peaks to be detected.
    min_distance : float
        The minimum distance between peaks to be detected.
    prominence : float
        The prominence of the peaks to be detected.
    calibrator : bool
        If True, the peak detection will be performed on the MP_CALIBRATOR instance instead of the MP_ANALYZER instance.
    experiment : str
        The name of the experiment to perform peak detection on.
        If 'all', peak detection will be performed on all experiments.
    
    Returns
    -------
    str
        JSON string with peak positions for each experiment.
    """

    py_object = MP_CALIBRATOR if calibrator else MP_ANALYZER

    results = {}

    for model_name in py_object.models:

        if experiment != 'all' and model_name != experiment:
            continue

        model = py_object.models[model_name]

        if model.hist_counts is None:
            raise ValueError(
                f"Histogram for model {model_name} has not been created."
            )

        model.guess_peaks(
            min_height=min_height,
            min_distance=min_distance,
            prominence=prominence
        )

        results[model_name] = model.peaks_guess.tolist()

    return json.dumps(results)

@tool_with_log()
def fit_multi_gaussian(
    peaks_guess: list[float] | dict[str, list[float]] | None = None,
    mean_tolerance: float | None = None,
    std_tolerance: float | None = None,
    threshold: float | None = None,
    fit_baseline: bool = False,
    baseline: float = 0.0,
    min_height: float = 10,
    min_distance: float = 4,
    prominence: float = 4,
    calibrator: bool = False,
    experiment: str = 'all') -> str:
    """
    Fit a multi-gaussian model to the histograms.

    Parameters
    ----------
    peaks_guess : list[float] | None
        A list of initial guesses for the peaks of the gaussians.
        If a dictionary is supplied, keys must match experiment names and values
        must be lists of initial guesses for the peaks of the gaussians for each
        experiment.
    mean_tolerance : float | None
        The tolerance for the mean of the gaussians.
        If None, defaults will be applied: guess ± abs(guess)/2
    std_tolerance : float | None
        The tolerance for the standard deviation of the gaussians.
        If None, the maximum fitted standard deviation will be equal to the initial guesses.
    threshold : float | None
        For masses: minimum value that can be observed (in kDa units). Default is 40.
        For contrasts: maximum value that can be observed (should be negative). Default is -0.0024.
        If None, defaults are applied based on detected data type.
    fit_baseline : bool
        If True, a baseline will be fitted and subtracted from the histograms before fitting the gaussians.
        The baseline argument will be ignored in this case.
    baseline : float
        The baseline value to be subtracted from the histograms before fitting.
        Useful to remove background noise.
    min_height : float
        The minimum height of the peaks to find the initial guesses.
    min_distance : float
        The minimum distance between the peaks to find the initial guesses.
    prominence : float
        The prominence of the peaks to find the initial guesses.
    calibrator : bool
        If True, the fitting will be performed on the MP_CALIBRATOR instance instead of the MP_ANALYZER instance.
    experiment : str
        The name of the experiment to fit the model to.
        If 'all', the fitting will be performed on all experiments.

    Returns
    -------
    str
        A message indicating the success or failure of the fitting process.

    Notes
    -----
    After running this tool, we recommend running ``show_fitted_parameters()``
    to see the results, and running ``get_legends_dataframe()`` to obtain the
    legends information before plotting with ``plot_histograms_and_fits()``.
    """

    py_object = MP_CALIBRATOR if calibrator else MP_ANALYZER

    assert isinstance(peaks_guess, (list, dict, type(None)))
    assert isinstance(mean_tolerance, (int, float, type(None)))
    assert isinstance(std_tolerance, (int, float, type(None)))
    assert isinstance(threshold, (int, float, type(None)))
    assert isinstance(baseline, (int, float))
    assert isinstance(min_height, (int, float))
    assert isinstance(min_distance, (int, float))
    assert isinstance(prominence, (int, float))

    for model_name in list(py_object.models.keys()):

        if experiment != 'all' and model_name != experiment:
            continue

        model = py_object.models[model_name]

        # Verify that we have created the histogram
        # and raise an error if we have not done it
        # Suggest running the `create_histogram_automatic()` or `create_histogram_manual()` method
        if model.hist_counts is None:
            message = f"Histogram for model {model_name} has not been created. Please run `create_histogram()` first."
            raise ValueError(message)

        # Determine which peaks to use
        if peaks_guess is None:

            if not hasattr(model, "peaks_guess") or model.peaks_guess is None:
                model.guess_peaks(
                    min_height=min_height,
                    min_distance=min_distance,
                    prominence=prominence,
                )

            if model.peaks_guess is None:
                raise ValueError(
                    f"No peaks available for model {model_name}. "
                    "Run guess_peaks() first or provide peaks_guess."
                )

            peaks_guess_local = model.peaks_guess

        elif isinstance(peaks_guess, dict):

            if model_name not in peaks_guess:
                raise ValueError(
                    f"No peaks provided for experiment '{model_name}'."
                )

            peaks_guess_local = peaks_guess[model_name]

        else:

            peaks_guess_local = peaks_guess

        model.fit_histogram(
            peaks_guess=peaks_guess_local,
            mean_tolerance=mean_tolerance,
            std_tolerance=std_tolerance,
            threshold=threshold,
            fit_baseline=fit_baseline,
            baseline=baseline
        )

    return "Multi-gaussian fitting completed successfully."

@tool_with_log()
def show_fitted_parameters(
    calibrator: bool = False) -> str:
    """
    Show the fitted parameters of the multi-gaussian model.
    
    Parameters
    ----------
    calibrator : bool
        If True, the fitted parameters will be shown for the MP_CALIBRATOR instance instead of MP_ANALYZER.
        This is useful for calibration data.
    
    Returns
    -------
    str
        A JSON string containing the fitted parameters of the multi-gaussian model.
    """

    py_object = MP_CALIBRATOR if calibrator else MP_ANALYZER

    fit_tables = py_object.get_properties('fit_table')

    # For each table - add a column with the name of the model
    for i, fit_table in enumerate(fit_tables):
        fit_table['name'] = list(py_object.models.keys())[i]

    # Concatenate all fit tables into a single DataFrame
    all_fit_tables = pd.concat(fit_tables, ignore_index=True)

    # Return as JSON
    fit_tables_json = all_fit_tables.to_json(orient='records')

    return fit_tables_json

@tool_with_log()
def get_legends_dataframe(
    repeat_colors: bool = True,
    calibrator: bool = False,
) -> list[dict[str, Any]]:
    """
    Obtain default labels and colors to plot the fitted histograms.

    Parameters
    ----------
    repeat_colors : bool, default True
        If True, repeat the same color scheme for each model's peaks.
        If False, use sequential colors across all peaks from all models.
    calibrator : bool
        If True, obtain legends for the MP_CALIBRATOR instance instead of
        MP_ANALYZER.

    Returns
    -------
    list[dict[str, Any]]
        Legend configuration records with the fields ``legends``, ``color``,
        ``select``, and ``show_legend``.
    """
    py_object = MP_CALIBRATOR if calibrator else MP_ANALYZER

    legends_df, _ = py_object.create_plotting_config(
        repeat_colors=repeat_colors
    )

    # Return native structured data for MCP instead of JSON encoded as a string.
    return legends_df.to_dict(orient="records")

@tool_with_log()
def plot_histograms_and_fits(
    legends_df: list[dict[str, Any]] | str | None = None,
    colors_hist: list[str] | str | None = None,
    save_as_html: bool = False,
    calibrator: bool = False
) -> str:

    """
    Plot the histograms and fitted curves using the current plotting configuration.
    The plotting configuration is defined by the PLOT_CONFIG, LEGEND_CONFIG, LAYOUT_CONFIG, and AXIS_CONFIG.
    To update use the tools `update_plot_config`, `update_legend_config`, `update_layout_config`, and `update_axis_config`.

    Parameters  
    ----------
    legends_df : list[dict[str, Any]] | str | None
        Legend configuration records containing legends, colors, and selections.
        A legacy JSON string is also accepted for direct Python callers.
        If None, the default legends will be used.
        This DataFrame affects the fitted curves only, not the histograms.
        It contains the columns ['legends', 'color', 'select', 'show_legend'].
        The 'select' column is a boolean mask indicating which peaks to include in the plot.
        The 'show_legend' column controls if the corresponding legend is shown.
    colors_hist : list[str] | str | None
        A list of colors for the histograms (one per model).
        If a string, it will be used for all histograms,
        If None, default colors will be used.
    save_as_html : bool
        If True, the plot will be also saved as an HTML file.
    calibrator : bool
        If True, the histograms and fits will be plotted for the MP_CALIBRATOR instance instead of MP_ANALYZER.
        This is useful for calibration data.
    Returns
    -------
    str
        The path to the saved plot.
    Note
    ----
    The legends_df records can be obtained by calling the get_legends_dataframe tool.

    """

    if isinstance(legends_df, str):
        # Backward compatibility for direct Python callers that still provide
        # the pre-MCP-2 JSON-string representation.
        legends_df = json.loads(legends_df)

    if legends_df is not None:
        legends_df = pd.DataFrame(legends_df)

    py_object = MP_CALIBRATOR if calibrator else MP_ANALYZER

    # Verify if colors_hist is a string and convert to list of strings
    if isinstance(colors_hist, str):
        colors_hist = [colors_hist] * len(py_object.models)

    time_str = datetime.now().strftime('%H-%M-%S')

    if save_as_html:

        fig = pm_plot_histograms_and_fits(
            py_object,
            legends_df=legends_df,
            colors_hist=colors_hist,
            plot_config=PLOT_CONFIG,
            legend_config=LEGEND_CONFIG,
            layout_config=LAYOUT_CONFIG,
            axis_config=AXIS_CONFIG
        )

        html_path = os.path.join(DATA_DIR, f"histograms_and_fits_{time_str}.html")
        fig.write_html(html_path)

    # For a strange reason the padding of the y-axis label in HTML figures is larger 
    # than in static figures. So we re do the plot with larger spacing

    LAYOUT_CONFIG.extra_padding_y_label = LAYOUT_CONFIG.extra_padding_y_label + 0.02

    fig = pm_plot_histograms_and_fits(
        py_object,
        legends_df=legends_df,
        colors_hist=colors_hist,
        plot_config=PLOT_CONFIG,
        legend_config=LEGEND_CONFIG,
        layout_config=LAYOUT_CONFIG,
        axis_config=AXIS_CONFIG
    )
    
    # Reset back AXIS_CONFIG
    LAYOUT_CONFIG.extra_padding_y_label = LAYOUT_CONFIG.extra_padding_y_label - 0.02

    # Export it also with the PLOT_CONFIG.plot_type
    plot_type = PLOT_CONFIG.plot_type.lower()
    if plot_type in ['png', 'jpg', 'jpeg','svg']:
        image_path = os.path.join(DATA_DIR, f"histograms_and_fits_{time_str}.{plot_type}")
        fig.write_image(image_path,
                        width=PLOT_CONFIG.plot_width, 
                        height=PLOT_CONFIG.plot_height)
        return f"Histograms and fits plot saved as {plot_type} at {image_path}."
    else:
        return "Histograms and fits plot created successfully, but no valid image format was specified for saving."

@tool_with_log()
def calibrate(known_standards: list[float] | list[list[float]]) -> str:
    """
    Obtain a calibration function using a list of known standards
    The calibration function is defined as f(mass) = contrast
    f(mass) = slope * mass + intercept

    Parameters
    ----------
    known_standards : list[float] | list[list[float]]
        Known standard values to use for calibration.
        Can be a list of floats in we only have one MP file
        Must be a list of lists with floats if we have multiple MP files
    Returns
    -------
    str
        A json string with the results of the calibration
    Note
    ----
    An example input for known_standards is [[148, 66], [480]] if we have two MP files
    An example input for one MP file is [480,148,66]
    """

    assert isinstance(known_standards, list)

    # Each element in the list is also a list here called sublist
    # Each sublist contains known mass values, from highest to lowest
    # There is one sublist per model in MP_CALIBRATOR

    # Verify each known_standards is a list of lists
    if isinstance(known_standards[0], (int, float)):
        # If the first element is not a list, we have a single MP file
        known_standards = [known_standards]

    # Raise value error if length of known_standards is not equal to number of models
    if len(known_standards) != len(MP_CALIBRATOR.models):
        raise ValueError("Length of known_standards must match number of models.")

    # Flatten the standards
    known_standards = np.concatenate(known_standards)

    MP_CALIBRATOR.master_calibration(calibration_standards=known_standards)

    # Print the calibration results
    calibration_dic = MP_CALIBRATOR.calibration_dic

    slope = MP_CALIBRATOR.calibration_dic['fit_params'][0]
    intercept = MP_CALIBRATOR.calibration_dic['fit_params'][1]
    r2 = MP_CALIBRATOR.calibration_dic['fit_r2']

    # Assign the standards to the MP_CALIBRATOR instance
    MP_CALIBRATOR.known_standards = known_standards

    # Assign the calibration dic to the MP_CALIBRATOR instance
    MP_CALIBRATOR.calibration_dic = calibration_dic

    # return a message with the slope, intercept and R2
    return f"Calibration results: Slope = {slope}, Intercept = {intercept}, R2 = {r2}"

@tool_with_log()
def plot_calibration(save_as_html: bool) -> str:
    """
    Plot the calibration results.
    Parameters
    ----------
    save_as_html : bool
        Whether to save the plot as an HTML file.
    Returns
    -------
    str
        A message indicating the success or failure of the plotting process.
    """

    time_str = datetime.now().strftime('%H-%M-%S')

    fig = pm_plot_calibration(
        mass=MP_CALIBRATOR.known_standards,
        contrast=MP_CALIBRATOR.calibration_dic['exp_points'],
        slope=MP_CALIBRATOR.calibration_dic['fit_params'][0],
        intercept=MP_CALIBRATOR.calibration_dic['fit_params'][1],
        plot_config=PLOT_CONFIG,
        axis_config=AXIS_CONFIG)

    if save_as_html:

        html_path = os.path.join(DATA_DIR, f"calibration_{time_str}.html")
        fig.write_html(html_path)

    # Export it also with the PLOT_CONFIG.plot_type
    plot_type = PLOT_CONFIG.plot_type.lower()
    if plot_type in ['png', 'jpg', 'jpeg','svg']:
        image_path = os.path.join(DATA_DIR, f"calibration_{time_str}.{plot_type}")
        fig.write_image(image_path)
        return f"Calibration plot saved as {plot_type} at {image_path}."
    else:
        return "Calibration plot created successfully, but no valid image format was specified for saving."
