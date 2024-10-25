"""
Script to apply segmentation to a set of reconstructed, masked images.

Input arguments:
    --input: input directory (Must be BIDS compliant)
    --i_derivative: which derivative to use as input (default: 'niftymic', other 'nesvor')
    --output: output directory (name of derivatives in the original BIDS compliant directory)
    --alg: segmentation algorithm (default: 'synthseg', other 'dhcp')
    --subject: subject ID (if not specified, all subjects will be processed)
    --HPC: if specified, the script will be configured to run in the HPC cluster
Usage:
    python run_segmentation.py --input <input> --i_derivative <derivative> --output <output> --alg <alg> [--subject <subject>]
"""

import os
import argparse
import subprocess
import pandas as pd
import bids as pybids
import shutil
import json
import docker
import sys
import glob
import numpy as np
import nibabel as nib
from source.utils import get_gestational_weeks, create_description_file
from source.seg import copy_brain_mask

# global paths
SINGULARITY_DHCP = (
    "/home/gmarti/SINGULARITY/dhcp-structural-pipeline_fetal-custom.sif"
)

# global path docker
DOCKER_DHCP = "gerardmartijuan/dhcp-pipeline-multifact:latest"

env = {
    **os.environ,
}


def main(args):
    try:
        layout = pybids.BIDSLayout(args.input, validate=False)
        layout.add_derivatives(os.path.join(args.input, "derivatives"))
    except AssertionError:
        print("The input directory is not a BIDS directory, exiting...")
        sys.exit()

    # load the participants.tsv or .csv file
    participants_file = os.path.join(args.input, "participants")
    if os.path.exists(participants_file + ".tsv"):
        participants = pd.read_csv(participants_file + ".tsv", sep="\t")
    elif os.path.exists(participants_file + ".csv"):
        participants = pd.read_csv(participants_file + ".csv")
    else:
        raise FileNotFoundError(
            f"participants.tsv or participants.csv file not found at {args.input}"
        )
    output_suffix = (
        args.output_suffix
        if args.output_suffix[0] == "_"
        else f"_{args.output_suffix}"
    )
    # Create output directory as new derivative folder
    be_dir = os.path.join(
        args.input, "derivatives", args.i_derivative + output_suffix
    )
    os.makedirs(be_dir, exist_ok=True)

    # if not exists the description file, create it new, with
    # information about the algorithm used
    create_description_file(be_dir, "BOUNTI brain extraction", "docker")

    # get subject if specified, else, get all subjects
    if args.subject is not None:
        subjects = [args.subject]
    else:
        subjects = layout.get_subjects()
    for subject in subjects:
        sessions = layout.get_sessions(subject=subject)
        sessions = [None] if len(sessions) == 0 else sessions
        for ses in sessions:
            print("Processing subject: " + subject)

            sub_ses_folder = (
                f"sub-{subject}/ses-{ses}"
                if ses is not None
                else f"sub-{subject}"
            )
            sub_ses = (
                f"sub-{subject}_ses-{ses}"
                if ses is not None
                else f"sub-{subject}"
            )
            out_bids_dir = os.path.abspath(
                os.path.join(
                    args.input,
                    "derivatives",
                    args.i_derivative + output_suffix,
                    sub_ses_folder,
                    "anat",
                )
            )
            os.makedirs(out_bids_dir, exist_ok=True)

            if not os.path.exists(
                # Does this even work?
                f"{out_bids_dir}/*_T2w.nii.gz"
            ):
                if args.i_derivative != "raw":
                    recon_file = os.path.join(
                        args.input,
                        "derivatives",
                        args.i_derivative,
                        sub_ses_folder,
                        "anat",
                        f"{sub_ses}_T2w.nii.gz",
                    )
                    layout.get()

                    recons = layout.get(
                        scope="derivatives",
                        subject=subject,
                        session=ses,
                        suffix="T2w",
                        extension="nii.gz",
                        return_type="file",
                    )
                    if len(recons) == 0:
                        print(
                            "No reconstructions found for subject: " + subject
                        )
                        continue
                    elif len(recons) > 1:
                        recons = [
                            rec
                            for rec in recons
                            if os.path.join(
                                "derivatives",
                                args.i_derivative,
                                sub_ses_folder,
                            )
                            in rec
                        ]

                        res = "res-0p8"
                        recon_file = [
                            recon for recon in recons if res in recon
                        ]

                        print(
                            f"\tMultiple reconstructions found for subject: {subject}. Using {res} resolution"
                        )
                        if len(recon_file) > 1:
                            print(
                                f"\t\tToo many reconstructions matching found: {recon_file}. Skipping"
                            )
                            continue
                        recon_file = recon_file[0]
                    else:
                        recon_file = recons[0]
                else:
                    raise NotImplementedError
                if not os.path.exists(recon_file):
                    print("No reconstructions found for subject: " + subject)
                    continue

            # if force, remove the output directory
            if args.force:
                shutil.rmtree(out_bids_dir)

            os.makedirs(out_bids_dir, exist_ok=True)
            # check if the skull stripping is done
            recon_file_name = os.path.basename(recon_file)
            mask_out = os.path.join(out_bids_dir, recon_file_name).replace(
                "_T2w.nii.gz", "_mask.nii.gz"
            )
            if os.path.exists(mask_out):
                print("\tSkull stripping already done for subject: " + sub_ses)
                continue
            # Call the segmentation algorithm

            if not os.path.exists(os.path.join(out_bids_dir, recon_file_name)):
                shutil.copyfile(
                    recon_file,
                    os.path.join(out_bids_dir, recon_file_name),
                )

            # call dhcp
            print("Calling BOUNTI brain extraction")
            # Create a temporary directory with a unique suffix
            import time

            tmp_dir = os.path.abspath(f"tmp_{time.time()}")
            os.makedirs(tmp_dir, exist_ok=True)
            cmd = (
                "docker run --rm "
                f"-v {out_bids_dir}:/home/data "
                f"-v {tmp_dir}:/home/out "
                "fetalsvrtk/segmentation:general_auto_amd bash /home/auto-proc-svrtk/scripts/auto-brain-bet-segmentation-fetal.sh "
                f"/home/data/ /home/out"
            )
            os.system(cmd)
            # Copy the content of tmp_dir to out_bids_dir and rename it:
            out_name = os.path.join(
                tmp_dir,
                recon_file_name.replace(".nii.gz", "-mask-bet-1.nii.gz"),
            )

            # Shutil copy to out_bids_dir
            shutil.copyfile(out_name, mask_out)
            shutil.rmtree(tmp_dir)
            # Mask the original image

            im_ni = nib.load(os.path.join(out_bids_dir, recon_file_name))
            mask_ni = nib.load(mask_out)
            masked_data = im_ni.get_fdata() * mask_ni.get_fdata()
            im_ni = nib.Nifti1Image(masked_data, im_ni.affine, im_ni.header)
            nib.save(im_ni, os.path.join(out_bids_dir, recon_file_name))


if __name__ == "__main__":
    # Parse input arguments
    parser = argparse.ArgumentParser(
        description="Script to apply segmentation to a set of reconstructed, masked images."
    )
    parser.add_argument(
        "--input",
        type=str,
        help="input directory (Must be BIDS compliant)",
        required=True,
    )
    parser.add_argument(
        "--i_derivative",
        type=str,
        help="which derivative to use as input (default: niftymic, other nesvor)",
        required=False,
        default="niftymic",
    )
    parser.add_argument(
        "--output_suffix",
        default="_masked",
        type=str,
        help="output directory (name of derivatives in the original BIDS compliant directory)",
        required=True,
    )

    parser.add_argument(
        "--subject",
        type=str,
        help="subject ID (if not specified, all subjects will be processed)",
        required=False,
        default=None,
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="if specified, the script will overwrite any existing segmentation",
        required=False,
    )

    args = parser.parse_args()

    main(args)

#### EXAMPLES OF RUN
# python run_brain_extraction.py --input /media/gerard/HDD/MULTIFACT_DATA/ERANET/ERANEU_MULTIFACT --i_derivative nesvor --output nesvor-brain_extracted --subject 290
