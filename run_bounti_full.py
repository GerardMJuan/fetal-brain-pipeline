"""Script that runs the complete bounti pipeline, ranging from preprocessing to reconstruction to segmentation.

Contains code from run_segmentation.py and run_recon.py, and will also contain code from run_seg_twai.py
"""

import os
import argparse
import subprocess
import pandas as pd
import bids as pybids
import shutil
import docker
import sys
import glob
import numpy as np
import time
from source.utils import (
    remove_directory_contents,
    create_brain_masks,
    create_description_file,
    get_subjects_and_stacks,
    get_gestational_weeks,
    denoise_image,
    bias_correction,
    get_cropped_stack_based_on_mask,
)
from source.seg import copy_brain_mask

from source.recon import (
    get_tp,
    get_all_thickness,
)

SINGULARITY_SVRTK = "svrtk_general_auto_amd.sif"

env = {
    **os.environ,
}


def reconstruct_volume(input_dir, recon_dir, subject):
    """Reconstruct the volume of a subject using the SVRTK pipeline.

    Parameters
    ----------
    input_dir : str
        The directory containing the input data.
    recon_dir : str
        The directory where the reconstructed data will be saved.
    subject : str
        The ID of the subject to be reconstructed.
    """
    # tp = get_tp(input_dir)
    # print("TP: ", tp)
    os.makedirs(os.path.join(recon_dir, "tmp_proc"), exist_ok=True)
    precommand = f"--nv -u --bind /home/gmarti/mock_home:/home/gmarti --bind {recon_dir}:/home/data/ --bind {recon_dir}/tmp_proc:/home/tmp_proc"
    reconstruction_cmd = f"bash /home/auto-proc-svrtk/scripts/auto-brain-reconstruction.sh /home/data/input /home/data" # 1 {tp} 0.8 1"
    singularity_image = SINGULARITY_SVRTK

    # Time the reconstruction
    start_time = time.time()

    print(
        f"singularity exec {precommand} {singularity_image} {reconstruction_cmd}"
    )
    with open(
        os.path.join(recon_dir, f"svrtk_recon.log"),
        "a",
        encoding="utf-8",
    ) as f:
        # second run reconstruction
        subprocess.run(
            f"singularity exec {precommand} {singularity_image} {reconstruction_cmd}",
            stdout=f,
            stderr=f,
            shell=True,
            check=True,
        )
        end_time = time.time()
        print(f"Reconstruction took {end_time - start_time} seconds", file=f)


def bounti_segmentation(input_dir, output_dir, subject):
    """
    This function performs the segmentation of a subject's brain using the Bounti pipeline.

    Parameters
    ----------
    input_dir : str
        The directory containing the input data.
    output_dir : str
        The directory where the segmented data will be saved.
    subject : str
        The ID of the subject to be segmented.
    gestational_weeks : int
        The gestational weeks of the subject.
    """
    os.makedirs(os.path.join(input_dir, "tmp_proc"), exist_ok=True)
    pre_command = f"-u --nv --bind /home/gmarti/mock_home:/home/gmarti --bind {output_dir}:/home/data/ --bind {input_dir}/tmp_proc:/home/tmp_proc"
    singularity_image = SINGULARITY_SVRTK
    reconstruction_cmd = "bash /home/auto-proc-svrtk/scripts/auto-brain-bounti-segmentation-fetal.sh /home/data/input /home/data"
    # Time the reconstruction
    start_time = time.time()

    print(
        f"singularity exec {pre_command} {singularity_image} {reconstruction_cmd}"
    )
    with open(
        os.path.join(output_dir, f"svrtk_seg.log"),
        "a",
        encoding="utf-8",
    ) as f:
        # second run reconstruction
        subprocess.run(
            f"singularity exec {pre_command} {singularity_image} {reconstruction_cmd}",
            stdout=f,
            stderr=f,
            shell=True,
            check=True,
        )
        end_time = time.time()
        print(f"Reconstruction took {end_time - start_time} seconds", file=f)


def main(args):
    try:
        print("Reading BIDS layout...")
        layout = pybids.BIDSLayout(args.input)
    except AssertionError:
        print("The input directory is not a BIDS directory, exiting...")
        sys.exit()

    # Path to the directory containing the stacks
    input_dir = args.input
    output_name = args.output

    # Get subjects and stacks
    subjects = get_subjects_and_stacks(args, layout)

    # Iterate over subjects and stacks
    for subject in subjects:
        # get subject number stripping the sub- prefix (if its there) and to string
        subject = str(subject).lstrip("sub-")  # ("-")[1])

        # add two trailing zeros to subject number
        subject = subject.zfill(3)
        subject_number = subject

        # check if subject exists in the BIDS dataset
        if subject not in layout.get_subjects():
            print(f"Subject {subject} not in BIDS dataset, skipping...")
            continue

        # Get the runs
        list_of_runs = layout.get(
            scope="raw",
            subject=subject_number,
            target="run",
            return_type="id",
        )
        print("Total runs: ", len(list_of_runs))
        if args.subjects_csv is not None:
            # read the csv file
            df = pd.read_csv(args.subjects_csv)
            # get the row for the subject
            df = df.loc[df["sub"] == int(subject_number)]

            # Run the subject only if the rerun column is set to 1
            # if df["rerun"].values[0] != 1 and not args.only_preproc:
            #     print(f"Skipping subject {subject_number}, rerun not set to 1")
            #     continue

            # if there are bad slices (not empty)
            if not pd.isna(df["bad slices"].values[0]):
                # get the list of bad slices (it is separated by spaces, so split it)
                list_of_bad_slices = df["bad slices"].values[0].split(" ")

                # remove the bad slices from the list of runs
                list_of_runs = [
                    x for x in list_of_runs if x not in list_of_bad_slices
                ]
        print("Runs selected: ", len(list_of_runs))

        # Get the actual list of files
        # But only the files which its run number is in the list of runs
        list_of_files = []
        for run in list_of_runs:
            list_of_files += layout.get(
                scope="raw",
                subject=subject_number,
                run=run,
                extension=".nii.gz",
                return_type="file",
            )

        print(f"Processing subject {subject}")

        #### 00: Preprocessing
        # preproc_dir = os.path.join(
        #     input_dir,
        #     "derivatives",
        #     f"{output_name}_preproc",
        #     f"sub-{subject}",
        #     "ses-01",
        #     "anat",
        # )

        # # if the directory doesnt exist, create it
        # os.makedirs(preproc_dir, exist_ok=True)

        # # create a json file with the preprocessing parameters
        # create_description_file(
        #     os.path.join(input_dir, "derivatives", f"{output_name}_preproc"),
        #     f"{output_name}_preproc",
        # )

        # image_base_dir = os.path.join(preproc_dir, "input")
        # os.makedirs(image_base_dir, exist_ok=True)

        # list_of_denoise = []
        # for file in list_of_files:
        #     list_of_denoise.append(file.replace(".nii.gz", "_denoise.nii.gz"))

        # list_of_denoise = [
        #     x.replace(os.path.dirname(x), image_base_dir)
        #     for x in list_of_denoise
        # ]

        # denoise_list_exists = [os.path.exists(x) for x in list_of_denoise]
        # if not all(denoise_list_exists):
        #     print("Denoising using ANTs...")
        #     for image_path, image_out_path in zip(
        #         list_of_files, list_of_denoise
        #     ):
        #         denoise_image(image_path, image_out_path)

        # list_of_bias_denoise = []
        # for file in list_of_denoise:
        #     list_of_bias_denoise.append(
        #         file.replace("_denoise.nii.gz", "_bias_denoise.nii.gz")
        #     )

        # denoise_list_exists = [os.path.exists(x) for x in list_of_bias_denoise]
        # if not all(denoise_list_exists):
        #     print("Bias correction using N4...")
        #     for image_path, image_out_path in zip(
        #         list_of_denoise, list_of_bias_denoise
        #     ):
        #         bias_correction(image_path, image_out_path)

        # # Update parsing of bids layout
        # layout.add_derivatives(f"{input_dir}/derivatives/preproc")

        #### 01: Reconstruction

        recon_dir = os.path.join(
            input_dir,
            "derivatives",
            f"{output_name}_rec",
            f"sub-{subject}",
            "ses-01",
            "anat",
        )
        os.makedirs(recon_dir, exist_ok=True)

        # create a json file with the reconstruction parameters
        create_description_file(
            os.path.join(input_dir, "derivatives", f"{output_name}_rec"),
            f"{output_name}_rec",
        )

        input_recon_dir = os.path.join(recon_dir, "input")
        os.makedirs(input_recon_dir, exist_ok=True)

        # Check if the recon file exists
        recon_file = os.path.join(recon_dir, "reo-SVR-output-brain.nii.gz")
        if not os.path.exists(recon_file):
            print("Reconstructing volume...")

            # Copy the files to the recon directory
            for file in list_of_files:
                shutil.copy(file, input_recon_dir)

            reconstruct_volume(input_recon_dir, recon_dir, subject)

        #### 2: Segmentation

        # No need to get the gestational weeks here

        seg_dir = os.path.join(
            input_dir,
            "derivatives",
            f"{output_name}_seg",
            f"sub-{subject}",
            "ses-01",
            "anat",
        )

        os.makedirs(seg_dir, exist_ok=True)

        # create a json file with the segmentation parameters
        create_description_file(
            os.path.join(input_dir, "derivatives", f"{output_name}_seg"),
            f"{output_name}_seg",
        )

        input_seg_dir = os.path.join(seg_dir, "input")
        os.makedirs(input_seg_dir, exist_ok=True)

        # Check if the seg file exists
        seg_file = os.path.join(
            seg_dir, f"sub-{subject}_ses-01_T2w-mask-brain_bounti-19.nii.gz"
        )

        if not os.path.exists(seg_file):
            print("Segmenting volume...")

            # Copy the output recon file to the seg directory
            shutil.copy(recon_file, input_seg_dir)

            bounti_segmentation(input_seg_dir, seg_dir, subject)

        #### 3: Surface
        ## Cannot be done until we know how to segment the CC

        # surf_dir = os.path.join(
        #     input_dir,
        #     "derivatives",
        #     f"{output_name}_surf",
        #     f"sub-{subject}",
        #     "ses-01",
        #     "anat",
        # )

        # os.makedirs(surf_dir, exist_ok=True)

        # # create a json file with the segmentation parameters
        # create_description_file(
        #     os.path.join(input_dir, "derivatives", f"{output_name}_surf"), f"{output_name}_surf"
        # )

        # input_surf_dir = os.path.join(surf_dir, "input")
        # os.makedirs(input_surf_dir, exist_ok=True)


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
        "--output",
        type=str,
        required=True,
        help="Name of the derivatives folder created in the BIDS directory",
    )

    # argument to input the path to the csv file with the subjects and stacks
    parser.add_argument(
        "--subjects_csv",
        type=str,
        help="Path to the csv file with the information about qc to select the stacks to run for each subject. If not set, will run on all subjects and stacks",
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
