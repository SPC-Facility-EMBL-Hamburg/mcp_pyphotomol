import os
import json
from datetime import datetime
from ..config import DATA_DIR_NO_DATE
from ..mcp import mcp

@mcp.resource("data://{date}/logbook")
def get_mcp_logbook(date: str) -> str:

    """
    Retrieves the mcp server logbook for a specific date.
    Parameters
    ----------
    date : str
        The date for which the logbook is requested, formatted as 'YYYY-MM-DD' or 'DD-MM-YYYY'.
    Returns
    -------
    str
        JSON text with the logbook data for the specified date, or an error
        message if no logbook is found.
    """

    # Check if the date is in the correct format
    # If the date is in the format 'DD-MM-YYYY', convert it to 'YYYY-MM-DD'
    if '-' in date and len(date.split('-')[0]) == 2:
        day, month, year = date.split('-')
        date = f"{year}-{month}-{day}"

    # If an invalid datetime is provided, use today's date
    try:
        datetime.strptime(date, '%Y-%m-%d')
    except ValueError:
        date = datetime.today().strftime('%Y-%m-%d')
        
    # Can match data at
    # today = datetime.today().strftime('%Y-%m-%d')
    requested_dir = os.path.join(DATA_DIR_NO_DATE, date)
    if not os.path.isdir(requested_dir):
        return json.dumps({"error": "No logbook found for the specified date."})

    # Find the logbook file in the requested directory
    # By looking at the files with the pattern *mcp_logbook*

    logbook_files = [f for f in os.listdir(requested_dir) if 'mcp_logbook' in f]
    if not logbook_files:
        return json.dumps({"error": "No logbook found for the specified date."})

    # Read the json file and return it
    logbook_path = os.path.join(requested_dir, logbook_files[0])
    with open(logbook_path, 'r') as f:
        logbook_data = json.load(f)

    return json.dumps(logbook_data)
