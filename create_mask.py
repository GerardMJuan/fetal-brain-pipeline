"""
Small script to create a mask from a HD reconstructed file using MONAIfbs
"""

import os
import argparse
import subprocess
import pandas as pd
import bids as pybids
import sys
import shutil
import time
import glob
import json
import docker
from source.utils import (
    remove_directory_contents,
    create_brain_masks,
    create_description_file,
)
from source.utils import get_subjects_and_stacks, create_brain_masks
import sys

# create global variable with the name of the script
DOCKER_NIFTYMIC = ""
SINGULARITY_NIFTYMIC = ""


def main(args):
    HPC = args.HPC

    # Path to the directory containing the DICOM folders
    input_dir = args.input

    subject = args.subject
    derivative = args.derivative
    # check that the input_dir is a BIDS directory with the library pybids, if not, exit
    try:
        bids_layout = pybids.BIDSLayout(input_dir, derivatives=True)
    except AssertionError:
        print("The input directory is not a BIDS directory, exiting...")
        sys.exit()

    # Get subjects and stacks
    derivatives = bids_layout.get(
        scope=derivative,
        subject=subject,
        target="subject",
        return_type="file",
        suffix="T2w",
        extension="nii.gz",
    )
    print(derivatives)
    if len(derivatives) == 0:
        print(f"Subject {subject} is not reconstructed, skipping...")
        sys.exit()

    recon_file = derivatives[0]

    # name of the mask is same path as the recon file but with _mask.nii.gz
    mask_file = recon_file.replace("_T2w.nii.gz", "_mask.nii.gz")
    if not os.path.exists(mask_file):
        create_brain_masks(
            [recon_file],
            [mask_file],
            HPC,
            SINGULARITY_NIFTYMIC,
            DOCKER_NIFTYMIC,
        )


def parser_obj():
    # Path to the directory containing the DICOM folders
    parser = argparse.ArgumentParser(
        description="Run reconstruction algorithm on specific subjects or on all subjects"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the directory containing the patients. MUST be a BIDS compatible directory",
    )

    parser.add_argument(
        "--derivative",
        type=str,
        required=True,
        help="Name of the derivatives folder where the recon is",
    )

    # argument to input the path to the csv file with the subjects and stacks
    parser.add_argument(
        "--subjects_csv",
        type=str,
        help="Path to the csv file with the subjects and stacks to run the reconstruction on. If not set, will run on all subjects and stacks",
    )

    # flag that if set, will indicate we are running this in the HPC cluster
    parser.add_argument(
        "--HPC",
        action="store_true",
        default=False,
        help="If set, will run the reconstruction on the HPC cluster",
    )

    # list of str, optional, of the subjects to run the reconstruction on
    # if not set, will run on all subjects
    parser.add_argument(
        "--subject",
        type=str,
        help="Subject to run the reconstruction on. If not set, will run on all subjects. Format: XXX, only the number.",
    )

    return parser


if __name__ == "__main__":
    parser = parser_obj()
    args = parser.parse_args()

    main(args)
