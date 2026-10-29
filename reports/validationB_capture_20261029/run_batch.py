"""Compatibility entry for the archived trajectory-B capture; use data_collection/workflow.py for new sessions."""
from pathlib import Path
import runpy

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).resolve().parents[2] / 'data_collection/validationB_legacy.py'), run_name='__main__')
