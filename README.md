# fetal-brain-pipeline

Collection of scripts to process fetal MRI data, used for the MULTIFACT project.

## Base Directory Scripts

- `run_recon.py`: Main script for running reconstruction algorithms on fetal MRI data. Supports multiple reconstruction methods and handles BIDS-formatted datasets.

- `run_brain_extraction.py`: Performs brain extraction on fetal MRI data using specified algorithms.

- `run_bounti_full.py`: Executes the complete BOUNTI pipeline, including preprocessing, reconstruction, and segmentation steps.

## Source Directory

The `source/` directory contains utility functions and modules used by the main scripts:

- `utils.py`: Contains various utility functions for file handling, image processing, and BIDS-related operations.

- `recon.py`: Implements reconstruction-related functions, including thickness calculations and time point extraction.

- `seg.py`: Provides segmentation-related utilities, such as brain mask copying.

These modules are imported and used by the main scripts to perform specific tasks in the fetal brain processing pipeline.

## LICENSE

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
