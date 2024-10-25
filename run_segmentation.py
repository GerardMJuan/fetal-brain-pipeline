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
    input_dir = os.path.abspath(args.input)
    try:
        layout = pybids.BIDSLayout(input_dir, validate=False)
        layout.add_derivatives(os.path.join(input_dir, "derivatives"))
    except AssertionError:
        print("The input directory is not a BIDS directory, exiting...")
        sys.exit()

    # load the participants.tsv or .csv file
    participants_file = os.path.join(input_dir, "participants")
    if os.path.exists(participants_file + ".tsv"):
        participants = pd.read_csv(participants_file + ".tsv", sep="\t")
    elif os.path.exists(participants_file + ".csv"):
        participants = pd.read_csv(participants_file + ".csv")
    else:
        raise FileNotFoundError(
            f"participants.tsv or participants.csv file not found at {input_dir}"
        )

    # Depending on this parameter, script will run on singularity or docker
    HPC = args.HPC

    # container_type will be singularity if HPC; else, docker
    container_type = "singularity" if HPC else "docker"

    # Create output directory as new derivative folder
    seg_dir = os.path.join(input_dir, "derivatives", args.output)
    os.makedirs(seg_dir, exist_ok=True)

    # if not exists the description file, create it new, with
    # information about the algorithm used
    create_description_file(seg_dir, args.alg)

    # get subject if specified, else, get all subjects
    if args.subject is not None:

        subjects = args.subject

        # check if the gestational weeks is specified
        if args.ga is not None:
            gestational_weeks = args.ga
        else:
            gestational_weeks = None

    else:
        subjects = layout.get_subjects()

        # sort subjects
        subjects.sort()

        # sort participants by subject id, using the same order as subjects
        # use lambda function to remove the "sub-" prefix and convert to int
        participants = participants.sort_values(by=["participant_id"])

        subjects = np.array(subjects)
        gestational_weeks = None

    for subject in subjects:
        sessions = layout.get_sessions()
        sessions = [None] if len(sessions) == 0 else sessions
        for ses in sessions:

            print("Processing subject: " + subject)

            # read the csv file if it exists
            if args.subjects_csv is not None:
                df = pd.read_csv(args.subjects_csv, sep=",")
                # get the row corresponding to the subject
                row = df.loc[df["sub"] == int(subject)]
                # Check if column "recon_qc" is 1
                # if not, skip the subject
                if row["recon_qc"].values[0] != 1:
                    print("Reconstruction QC is not 1 for subject: " + subject)
                    continue

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
            out_bids_dir = os.path.join(
                input_dir,
                "derivatives",
                args.output,
                sub_ses_folder,
                "anat",
            )
            os.makedirs(out_bids_dir, exist_ok=True)

            if not os.path.exists(
                # Does this even work?
                f"{out_bids_dir}/segmentations/*_all_labels.nii.gz"
            ):
                if args.i_derivative != "raw":
                    recon_file = os.path.join(
                        input_dir,
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
                            if os.path.join("derivatives", args.i_derivative)
                            in rec
                        ]

                        res = "res-0p8"
                        recon_file = [
                            recon for recon in recons if res in recon
                        ]

                        print(
                            f"Multiple reconstructions found for subject: {subject}. Using {res} resolution"
                        )
                        if len(recon_file) > 1:
                            print(
                                f"\tToo many reconstructions matching found: {recon_file}. Skipping"
                            )
                            continue
                        print(recons)
                        recon_file = recon_file[0]
                    else:
                        recon_file = recons[0]
                else:
                    recon_file = os.path.join(
                        input_dir,
                        sub_ses_folder,
                        "anat",
                        f"{sub_ses}_rec-nmic_T2w.nii.gz",
                    )
                if not os.path.exists(recon_file):
                    print("No reconstructions found for subject: " + subject)
                    continue

            # if force, remove the output directory
            if args.force:
                shutil.rmtree(out_bids_dir)

            # check if the segmentation is done
            if args.seg and os.path.exists(
                f"{out_bids_dir}/segmentations/*_all_labels.nii.gz"
            ):
                print("Segmentation already done for subject: " + subject)
                args.seg = False

            if args.i_derivative == "raw":
                # we need to compute the mask using bet
                dseg_file = os.path.join(
                    input_dir,
                    f"sub-{subject}",
                    "anat",
                    f"sub-{subject}" + "_rec-nmic_dseg.nii.gz",
                )
                mask_file = os.path.join(
                    input_dir,
                    f"sub-{subject}",
                    "anat",
                    f"sub-{subject}" + "_ses-01_mask.nii.gz",
                )

            else:
                # we need to compute the mask using bet
                mask_file = os.path.join(
                    input_dir,
                    "derivatives",
                    args.i_derivative,
                    sub_ses_folder,
                    "anat",
                    f"{sub_ses}" + "_mask.nii.gz",
                )

            if not os.path.exists(mask_file) or args.force:
                # Remove existing recon_masked_file and mask_file if they exist and force

                print("Computing mask")

                if args.i_derivative != "raw":
                    # compute mask by calling fslmaths -bin
                    subprocess.run(
                        " ".join(
                            [
                                "fslmaths",
                                recon_file,
                                "-bin",
                                mask_file,
                            ]
                        ),
                        shell=True,
                        env=env,
                        check=True,
                    )
                else:
                    subprocess.run(
                        " ".join(
                            [
                                "fslmaths",
                                dseg_file,
                                "-bin",
                                mask_file,
                            ]
                        ),
                        shell=True,
                        env=env,
                        check=True,
                    )

            # get recon file without path and extension "extension is .nii.gz or .nii"
            recon_file_name = recon_file.split("/")[-1].split(".")[0]

            # get gestational weeks only if not specified
            if gestational_weeks is None:
                gestational_weeks = get_gestational_weeks(
                    subject, participants
                )

            if gestational_weeks is None:
                print(
                    "Gestational weeks not specified for subject: " + subject
                )
                continue

            # Call the segmentation algorithm
            if args.alg == "dhcp":
                print("Copying brain mask")
                copy_brain_mask(out_bids_dir, recon_file_name, mask_file)

                # call dhcp
                print("Calling Fetal DHCP")
                # if both --seg and --recon are specified, or neither, run the whole pipeline
                task = ""
                if args.seg == args.surf:
                    args.seg = True
                    args.surf = True
                    task = "-all"
                elif args.seg:
                    task = "-seg"
                elif args.surf:
                    task = "-surf"

                # threads are hardcoded to 4
                with open(
                    os.path.join(out_bids_dir, f"sub-{subject}_ses-1_seg.log"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    if HPC:
                        # first run segmentation
                        # maybe we need the number of tries
                        # run it with docker using the docker sdk
                        tries = 0
                        max_tries = 5 if args.seg else 1
                        while tries < max_tries and (
                            (
                                len(
                                    glob.glob(
                                        f"{out_bids_dir}/segmentations/*_all_labels.nii.gz"
                                    )
                                )
                                == 0
                                and args.seg
                            )
                            or args.surf
                        ):
                            string_to_run = " ".join(
                                [
                                    "singularity",
                                    "exec",
                                    "--env",
                                    "PREPEND_PATH=/home/gmarti/LIB/fsl/bin",  # hardcoded
                                    SINGULARITY_DHCP,
                                    "/usr/local/src/structural-pipeline/fetal-pipeline.sh",
                                    recon_file,
                                    str(gestational_weeks),
                                    "-data-dir",
                                    out_bids_dir,
                                    "-t",
                                    "1",
                                    "-c",
                                    "0",
                                    task,
                                ]
                            )

                            print("Try number: " + str(tries))
                            print(string_to_run)
                            result = subprocess.run(
                                string_to_run,
                                shell=True,
                                env=env,
                                check=True,
                                stderr=f,
                                stdout=f,
                            )
                            tries += 1

                    else:
                        # copy recon_file to the output directory
                        shutil.copyfile(
                            recon_file,
                            os.path.join(
                                out_bids_dir,
                                f"{recon_file_name}.nii.gz",
                            ),
                        )

                        # run it with docker using the docker sdk
                        tries = 0
                        max_tries = 5 if args.seg else 1
                        while tries < max_tries and (
                            (
                                len(
                                    glob.glob(
                                        f"{out_bids_dir}/segmentations/*_all_labels.nii.gz"
                                    )
                                )
                                == 0
                                and args.seg
                            )
                            or args.surf
                        ):

                            print("Try number: " + str(tries))
                            # command
                            command = [
                                f"/data/{os.path.basename(recon_file)}",
                                str(gestational_weeks),
                                "-data-dir",
                                ".",
                                "-t",
                                "8",
                                "-c",
                                "0",
                                task,
                            ]

                            client = docker.from_env()
                            container = client.containers.run(
                                DOCKER_DHCP,
                                user=os.getuid(),
                                command=command,
                                volumes={
                                    out_bids_dir: {
                                        "bind": "/data",
                                        "mode": "rw",
                                    },
                                },
                                detach=True,
                                # stderr=True,
                                # stdout=True,
                            )
                            for line in container.logs(stream=True):
                                print(line.strip())

                            # container.wait()
                            # save the logs to f
                            f.write(
                                container.logs(
                                    stdout=True, stderr=True
                                ).decode("utf-8")
                            )
                            container.remove()
                            tries += 1

            else:
                print("Unknown algorithm")
                sys.exit(0)


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
        "--output",
        type=str,
        help="output directory (name of derivatives in the original BIDS compliant directory)",
        required=True,
    )

    # argument to input the path to the csv file with the subjects and stacks
    parser.add_argument(
        "--subjects_csv",
        type=str,
        help="Path to the csv with information about qc.",
    )

    parser.add_argument(
        "--alg",
        type=str,
        help="segmentation algorithm (default: dhcp)",
        required=False,
        default="dhcp",
    )
    parser.add_argument(
        "--subject",
        type=str,
        help="subject ID (if not specified, all subjects will be processed)",
        required=False,
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--ga",
        type=float,
        help="gestational age in weeks (if not specified, it will be retrieved from the participants.tsv file)",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--HPC",
        action="store_true",
        help="if specified, the script will be configured to run in the HPC cluster",
        required=False,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="if specified, the script will overwrite any existing segmentation",
        required=False,
    )

    # only segmentation
    parser.add_argument(
        "--seg",
        action="store_true",
        help="if specified, the script will only run the segmentation part of the dhcp pipeline",
        required=False,
    )

    # only reconstruction
    parser.add_argument(
        "--surf",
        action="store_true",
        help="if specified, the script will only run the surface reconstruction part of the dhcp pipeline. Requires to have run the segmentation before",
        required=False,
    )

    args = parser.parse_args()

    main(args)

#### EXAMPLES OF RUN
# python run_segmentation.py --input /media/gerard/HDD/MULTIFACT_DATA/ERANET/ERANEU_MULTIFACT --i_derivative nesvor --output nesvor-dhcp-test --alg dhcp --subject 290 --surf

# python run_segmentation.py --input /media/gerard/HDD/MULTIFACT_DATA/CMV_BIDS --i_derivative niftymic --output dhcp_niftymic_test --alg dhcp --subject 042 --surf
