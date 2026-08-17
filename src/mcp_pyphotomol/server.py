import pyphotomol

from .config import DATA_DIR, DATA_DIR_NO_DATE, EXAMPLE_DATA_DIR, LOGBOOK_FILE


# Instance to handle the mass photometry count data.
MP_ANALYZER = pyphotomol.MPAnalyzer()

# Instance to handle the mass photometry calibration data.
MP_CALIBRATOR = pyphotomol.MPAnalyzer()

# Options for plotting.
PLOT_CONFIG = pyphotomol.PlotConfig(plot_height=800)
LEGEND_CONFIG = pyphotomol.LegendConfig()
LAYOUT_CONFIG = pyphotomol.LayoutConfig()
LAYOUT_CONFIG.stacked = True
AXIS_CONFIG = pyphotomol.AxisConfig()