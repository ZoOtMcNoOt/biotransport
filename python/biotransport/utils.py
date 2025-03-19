"""
Utility functions for BioTransport simulations.
"""

import os
import time

def get_results_dir(subfolder=None):
    """
    Get the path to the results directory, creating it if it doesn't exist.

    Args:
        subfolder: Optional subfolder name within results directory

    Returns:
        str: Path to the results directory
    """
    # Start with 'results' in the current directory
    results_dir = os.path.join(os.getcwd(), 'results')

    # Create with timestamp subfolder if requested
    if subfolder == 'timestamp':
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        results_dir = os.path.join(results_dir, timestamp)
    elif subfolder:
        results_dir = os.path.join(results_dir, subfolder)

    # Create the directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)

    return results_dir

def get_result_path(filename, subfolder=None):
    """
    Get the full path for a result file in the results directory.

    Args:
        filename: Name of the file
        subfolder: Optional subfolder within results directory

    Returns:
        str: Full path to the result file
    """
    return os.path.join(get_results_dir(subfolder), filename)