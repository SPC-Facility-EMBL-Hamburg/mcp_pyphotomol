import os
from datetime import datetime

from mcp_pyphotomol.paths import get_example_data_root, get_user_data_root


SKIP_USER_DATA_INIT = (
    os.environ.get("MCP_PYPHOTOMOL_SKIP_USER_DATA_INIT") == "1"
)

# Data directories
DATA_DIR_NO_DATE = str(get_user_data_root())
EXAMPLE_DATA_DIR = str(get_example_data_root())

if not SKIP_USER_DATA_INIT:
    os.makedirs(DATA_DIR_NO_DATE, exist_ok=True)

# Create a dated session directory.
today = datetime.today().strftime("%Y-%m-%d")
DATA_DIR = os.path.join(DATA_DIR_NO_DATE, today)

if not SKIP_USER_DATA_INIT:
    os.makedirs(DATA_DIR, exist_ok=True)

# Create the MCP logbook.
LOGBOOK_FILE = os.path.join(DATA_DIR, "mcp_logbook.txt")

if not SKIP_USER_DATA_INIT and not os.path.exists(LOGBOOK_FILE):
    with open(LOGBOOK_FILE, "w") as f:
        f.write("MCP Logbook\n")
        f.write(f"Date: {today}\n")
        f.write("MCP function calls will be added here.\n")


def build_server_instructions(data_dir: str) -> str:
    """Return MCP server instructions including the active output folder."""
    return (
        "This server provides tools for analysing mass photometry count data.\n"
        "You can import data, create and fit histograms with a multi-gaussian "
        "model, and plot the results.\n"
        "There are two important instances: MP_ANALYZER for analysis and "
        "MP_CALIBRATOR for calibration.\n"
        f"Plots and log files for this session are saved in: {data_dir}"
    )


SERVER_INSTRUCTIONS = build_server_instructions(DATA_DIR)