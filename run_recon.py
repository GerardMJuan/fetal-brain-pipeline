"""
Base file to run the reconstruction algorithm on the data

The input data SHOULD be in BIDS format.

The output data will be in a separate folder, and, if prompted, it will save only the result
in BIDS format, in the original folder.

The reconstruction algorithm is located in the same directory as this file, and it is named docker_recon.sh

Other features of the script are:
- If the subject is already done, it will skip it
- Can input a list of subjects to run the reconstruction on
- Optionally, can input the stacks (runs) to run the reconstruction on (default is all)
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
import nibabel
from source.utils import (
    remove_directory_contents,
    create_brain_masks,
    create_description_file,
    get_subjects_and_stacks,
    denoise_image,
    get_cropped_stack_based_on_mask,
)

from source.recon import (
    get_tp,
    get_all_thickness,
)

# create global variable with the name of the script
DOCKER_SVRTK = "fetalsvrtk/svrtk:auto-2.20"
DOCKER_NIFTYMIC = "gerardmartijuan/niftymic.multifact"
DOCKER_NESVOR = "junshenxu/nesvor:latest"

SINGULARITY_NIFTYMIC = "/home/gmarti/SINGULARITY/niftymic_upf_latest.sif"
SINGULARITY_NESVOR = "/home/gmarti/SINGULARITY/nesvor_latest.sif"
SINGULARITY_SVRTK = "/home/gmarti/SINGULARITY/fetalsvrtk_auto-2.20_fixed.sif"


def reconstruct_volume(
    input_recon_dir, mask_recon_dir, recon_dir, subject, algorithm, HPC
):
    """
    Reconstructs the volume using the specified algorithm
    And in the specified environment.
    """
    list_of_files = glob.glob(os.path.join(input_recon_dir, "*.nii.gz"))
    list_of_masks = glob.glob(os.path.join(mask_recon_dir, "*.nii.gz"))
    if HPC:
        # prepare the script to run on the HPC
        if algorithm == "niftymic":
            # then run the reconstruction
            precommand = "--nv"
            reconstruction_cmd = f"niftymic_run_reconstruction_pipeline --filenames {' '.join(str(x) for x in sorted(list_of_files))} --filenames-masks {' '.join(str(x) for x in sorted(list_of_masks))} --dir-output {recon_dir} --isotropic-resolution 0.8 --suffix-mask _mask --alpha 0.01 --automatic-target-stack 1 --run-bias-field-correction 1"  # changed from 0
            singularity_image = SINGULARITY_NIFTYMIC

        if algorithm == "nesvor":
            precommand = "--nv"
            # create folders for the slices and sim_slices
            # os.makedirs(os.path.join(recon_dir, "slices"), exist_ok=True)
            # os.makedirs(os.path.join(recon_dir, "sim_slices"), exist_ok=True)
            reconstruction_cmd = f"nesvor reconstruct --input-stacks {' '.join(str(x) for x in sorted(list_of_files))} --stack-masks {' '.join(str(x) for x in sorted(list_of_masks))} --registration svort-only --output-volume {recon_dir}/nesvor.nii.gz --output-model {recon_dir}/nesvor.pt --output-resolution 0.8 --n-levels-bias 1 --bias-field-correction"
            singularity_image = SINGULARITY_NESVOR

        if algorithm == "svrtk":
            # tp = get_tp(input_recon_dir)
            print("TP: ", tp)
            os.makedirs(os.path.join(recon_dir, "tmp_proc"), exist_ok=True)
            precommand = f"-u --bind /home/gmarti/mock_home:/home/gmarti --bind {recon_dir}:/home/data/ --bind {recon_dir}/tmp_proc:/home/tmp_proc"
            reconstruction_cmd = f"bash /home/auto-proc-svrtk/auto-brain-reconstruction.sh /home/data/input /home/data 1 {tp} 0.8 1"
            singularity_image = SINGULARITY_SVRTK

        # Time the reconstruction
        start_time = time.time()

        print(
            f"singularity exec {precommand} {singularity_image} {reconstruction_cmd}"
        )
        with open(
            os.path.join(recon_dir, f"{algorithm}_recon.log"),
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
            print(
                f"Reconstruction took {end_time - start_time} seconds", file=f
            )

    else:
        if algorithm == "niftymic":
            docker_image = DOCKER_NIFTYMIC
            docker_command = [
                "niftymic_run_reconstruction_pipeline",
                "--filenames",
                " ".join(
                    os.path.join("/input", os.path.basename(x))
                    for x in sorted(list_of_files)
                ),
                "--filenames-masks",
                " ".join(
                    os.path.join("/masks", os.path.basename(x))
                    for x in sorted(list_of_masks)
                ),
                "--dir-output",
                "/srr",
                "--isotropic-resolution",
                "0.8",
                "--suffix-mask",
                "_mask" "--alpha" "0.01",
                "--automatic-target-stack",
                "1",
                "--run-bias-field-correction",
                "1",
            ]
            docker_command = " ".join(docker_command)
            docker_volumes = {
                recon_dir: {"bind": "/srr", "mode": "rw"},
                input_recon_dir: {"bind": "/input", "mode": "rw"},
                mask_recon_dir: {"bind": "/masks", "mode": "rw"},
            }

        elif algorithm == "nesvor":
            docker_image = DOCKER_NESVOR
            docker_command = [
                "nesvor",
                "reconstruct",
                "--input-stacks",
                " ".join(str(x) for x in sorted(list_of_files)),
                "--stack-masks",
                " ".join(str(x) for x in sorted(list_of_masks)),
                "--output-volume",
                "/out/nesvor.nii.gz",
                "--output-resolution",
                "0.8",
                "--bias-field-correction",
                "--n-levels-bias",
                "1",
                "--batch-size",
                "8192",
            ]
            docker_command = " ".join(docker_command)
            docker_volumes = {
                recon_dir: {"bind": "/out", "mode": "rw"},
                input_recon_dir: {"bind": "/data", "mode": "rw"},
                mask_recon_dir: {"bind": "/data", "mode": "rw"},
            }

        elif algorithm == "svrtk":
            docker_image = DOCKER_SVRTK
            docker_command = [
                "bash",
                "/home/auto-proc-svrtk/auto-brain-reconstruction.sh",
                "/home/data/input",
                "/home/data",
            ]
            docker_command = " ".join(docker_command)
            docker_volumes = {
                recon_dir: {"bind": "/home/data", "mode": "rw"},
            }

        # use the docker interface for python
        print(f"Running {algorithm} reconstruction")
        client = docker.from_env()

        if algorithm != "nesvor":
            # no need for gpu
            gpu = []
        else:
            gpu = [
                docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
            ]

        container = client.containers.run(
            docker_image,
            user=os.getuid(),
            command=docker_command,
            volumes=docker_volumes,
            detach=True,
            device_requests=gpu,
            stderr=True,
            stdout=True,
        )
        container.wait()
        # print the logs
        print(container.logs().decode("utf-8"))
        container.remove()

    # for each algorithm, copy the result to the output folder
    # with the correct BIDS name
    if algorithm == "niftymic":
        result_img = f"{recon_dir}/recon_template_space/srr_template.nii.gz"
        result_mask = (
            f"{recon_dir}/recon_template_space/srr_template_mask.nii.gz"
        )
    elif algorithm == "nesvor":
        result_img = f"{recon_dir}/nesvor.nii.gz"
        result_mask = None
    elif algorithm == "svrtk":
        result_img = f"{recon_dir}/reo-SVR-output-brain.nii.gz"
        result_mask = None

    bids_output_recon = os.path.join(
        recon_dir,
        f"sub-{subject}_ses-01_T2w.nii.gz",
    )
    shutil.copy(result_img, bids_output_recon)

    # if the mask exists, copy it too
    if result_mask is not None and os.path.exists(result_mask):
        bids_output_mask = os.path.join(
            recon_dir,
            f"sub-{subject}_ses-01_mask.nii.gz",
        )
        shutil.copy(result_mask, bids_output_mask)


def main(args):
    HPC = args.HPC

    # Path to the directory containing the DICOM folders
    input_dir = args.input

    # Path to the output directory
    output_name = args.output

    algorithm = args.algorithm

    # check that the input_dir is a BIDS directory with the library pybids, if not, exit
    try:
        print("Reading BIDS layout...")
        bids_layout = pybids.BIDSLayout(
            input_dir,
        )
        bids_layout.add_derivatives(f"{input_dir}/derivatives/{output_name}")
        print("Done!")
    except AssertionError:
        print("The input directory is not a BIDS directory, exiting...")
        sys.exit()

    # Get subjects and stacks
    subjects = get_subjects_and_stacks(args, bids_layout)

    # iterate over the subjects and stacks
    for subject in subjects:
        # get subject number stripping the sub- prefix (if its there) and to string
        subject = str(subject).lstrip("sub-")  # ("-")[1])

        # add two trailing zeros to subject number
        subject = subject.zfill(3)
        subject_number = subject

        # check if subject exists in the BIDS dataset
        if subject not in bids_layout.get_subjects():
            print(f"Subject {subject} not in BIDS dataset, skipping...")
            continue

        # check if inside the derivatives folder we have the result, using pybids
        # if we have the result, skip the subject
        derivatives = bids_layout.get(
            scope=output_name,
            subject=subject,
            target="subject",
            extension="nii.gz",
            suffix="T2w",
            return_type="file",
        )
        print(derivatives)
        if len(derivatives) > 0 and not args.force and not args.only_preproc:
            print(f"Subject {subject} already reconstructed, skipping...")
            continue

        print(f"Processing subject {subject}")

        # get the list of runs for that subject
        list_of_runs = bids_layout.get(
            scope="raw",
            subject=subject_number,
            target="run",
            return_type="id",
        )
        print("Total runs: ", len(list_of_runs))
        # read the csv file if it exists
        if args.subjects_csv is not None:
            # read the csv file
            df = pd.read_csv(args.subjects_csv, sep="\t")
            # get the row for the subject
            df = df.loc[df["participant_id"] == f"sub-{subject_number}"]

            # # Run the subject only if the rerun column is set to 1
            # if df["rerun"].values[0] != 1 and not args.only_preproc:
            #     print(f"Skipping subject {subject_number}, rerun not set to 1")
            #     continue

            # if hte algorithm is niftymic
            if algorithm == "niftymic":
                # if column "niftymic_qc" is set to 1, skip the subject
                if df["niftymic_qc"].values[0] == 1:
                    print(
                        f"Skipping subject {subject_number}, niftymic_qc set to 1"
                    )
                    continue

            # if there are bad slices (not empty)
            if not pd.isna(df["bad slices"].values[0]):
                # get the list of bad slices (it is separated by spaces, so split it)
                list_of_bad_slices = df["bad slices"].values[0].split(" ")

                # remove the bad slices from the list of runs
                list_of_runs = [
                    x for x in list_of_runs if x not in list_of_bad_slices
                ]
        print("Runs selected: ", len(list_of_runs))

        list_of_files = bids_layout.get(
            scope="raw",
            subject=subject_number,
            target="subject",
            return_type="file",
            extension="nii.gz",
        )

        # directory where the preprocessed data will be held
        preproc_dir = os.path.join(
            input_dir,
            "derivatives",
            "preproc",
            f"sub-{subject}",
            "ses-01",
            "anat",
        )

        # if os.path.exists(preproc_dir) and args.force:
        #    remove_directory_contents(preproc_dir)

        # if the directory doesnt exist, create it
        os.makedirs(preproc_dir, exist_ok=True)

        mask_recon_dir = os.path.join(preproc_dir, "masks")
        os.makedirs(mask_recon_dir, exist_ok=True)

        image_base_dir = os.path.join(preproc_dir, "input")
        os.makedirs(image_base_dir, exist_ok=True)

        list_of_denoise = []
        for file in list_of_files:
            list_of_denoise.append(file.replace(".nii.gz", "_denoise.nii.gz"))

        list_of_denoise = [
            x.replace(os.path.dirname(x), image_base_dir)
            for x in list_of_denoise
        ]

        # Create the masks if they don't exist
        # mask should be equivalent to the input files in list_of_files but with _mask added
        list_of_masks = []
        for file in list_of_files:
            list_of_masks.append(file.replace(".nii.gz", "_mask.nii.gz"))

        list_of_masks_base = []
        for file in list_of_files:
            list_of_masks_base.append(
                file.replace(".nii.gz", "_maskbase.nii.gz")
            )

        # change the list of masks to the mask_recon_dir
        list_of_masks_base = [
            x.replace(os.path.dirname(x), mask_recon_dir)
            for x in list_of_masks_base
        ]

        # change the list of masks to the mask_recon_dir
        list_of_masks = [
            x.replace(os.path.dirname(x), mask_recon_dir)
            for x in list_of_masks
        ]

        # RUN PREPROC
        # check if any of those masks exist, if not, create them
        mask_list_exists = [os.path.exists(x) for x in list_of_masks_base]
        if not all(mask_list_exists):
            print("Creating masks using NiftyMIC...")
            create_brain_masks(
                list_of_files,
                list_of_masks_base,
                HPC,
                SINGULARITY_NIFTYMIC,
                DOCKER_NIFTYMIC,
            )

        ### DENOISE
        denoise_list_exists = [os.path.exists(x) for x in list_of_denoise]
        if not all(denoise_list_exists):
            print("Denoising using ANTs...")
            for image_path, image_out_path in zip(
                list_of_files, list_of_denoise
            ):
                denoise_image(image_path, image_out_path)

        # change the list of inputs to the input_recon_dir
        list_of_crops = [
            x.replace(os.path.dirname(x), preproc_dir) for x in list_of_files
        ]

        # check again, if any of those masks exist, if not, exit
        mask_list_exists = [os.path.exists(x) for x in list_of_masks_base]
        crops_exists = [os.path.exists(x) for x in list_of_crops]
        denoise_exists = [os.path.exists(x) for x in list_of_denoise]
        if all(mask_list_exists) and all(denoise_exists):
            # if all the crops exists, no need to run the preprocessing
            if not all(crops_exists):
                # for each stack, crop it based on mask
                for (
                    image_path,
                    mask_path,
                    mask_out_path,
                    image_out_path,
                ) in zip(
                    list_of_denoise,
                    list_of_masks_base,
                    list_of_masks,
                    list_of_crops,
                ):
                    if algorithm != "svrtk":
                        # crop the image based on the mask
                        image_ni = nibabel.load(image_path)
                        mask_ni = nibabel.load(mask_path)

                        # if mask is empty, skip it
                        if mask_ni.get_fdata().sum() == 0:
                            print("Empty mask, skipping...")
                            nibabel.save(image_ni, image_out_path)
                            nibabel.save(mask_ni, mask_out_path)  # overwrite
                            continue

                        image_cropped = get_cropped_stack_based_on_mask(
                            image_ni, mask_ni
                        )
                        mask_cropped = get_cropped_stack_based_on_mask(
                            mask_ni, mask_ni
                        )

                        # apply the mask to the image
                        image_cropped_masked = (
                            image_cropped.get_fdata()
                            * mask_cropped.get_fdata()
                        )

                        # update the image with the cropped image
                        image_cropped = nibabel.Nifti1Image(
                            image_cropped_masked, image_cropped.affine
                        )

                        # and save it appropiately
                        nibabel.save(image_cropped, image_out_path)
                        nibabel.save(mask_cropped, mask_out_path)  # overwrite

            else:
                print("Skipping preprocessing, crops already exist...")

        else:
            print("Error creating masks, exiting...")
            sys.exit()

        # Update parsing of bids layout
        bids_layout.add_derivatives(f"{input_dir}/derivatives/preproc")

        ## RUN RECONSTRUCTION
        if not args.only_preproc:
            # directory where the recon will be saved
            recon_dir = os.path.join(
                input_dir,
                "derivatives",
                output_name,
                f"sub-{subject}",
                "ses-01",
                "anat",
            )

            # remove contents of output folder
            # if os.path.exists(recon_dir) and args.force:
            #     remove_directory_contents(recon_dir)

            os.makedirs(recon_dir, exist_ok=True)

            mask_recon_dir = os.path.join(recon_dir, "masks")
            os.makedirs(mask_recon_dir, exist_ok=True)

            input_recon_dir = os.path.join(recon_dir, "input")
            os.makedirs(input_recon_dir, exist_ok=True)

            # copy the files to the input_recon_dir that are in the list of good files
            print(list_of_runs)
            for run_id in list_of_runs:
                # get the file

                if algorithm == "svrtk":
                    files = bids_layout.get(
                        scope="preproc",
                        subject=subject_number,
                        run=run_id,
                        suffix="denoise",
                        target="subject",
                        return_type="file",
                    )
                else:
                    files = bids_layout.get(
                        scope="preproc",
                        subject=subject_number,
                        run=run_id,
                        suffix="T2w",
                        target="subject",
                        return_type="file",
                    )

                print(files)

                # copy it to the input_recon_dir
                shutil.copy(files[0], input_recon_dir)

                if algorithm == "svrtk":
                    # if svrtk, copy the mask too
                    files = bids_layout.get(
                        scope="preproc",
                        subject=subject_number,
                        run=run_id,
                        suffix="maskbase",
                        target="subject",
                        return_type="file",
                    )
                else:
                    # copy the mask too
                    files = bids_layout.get(
                        scope="preproc",
                        subject=subject_number,
                        run=run_id,
                        suffix="mask",
                        target="subject",
                        return_type="file",
                    )
                shutil.copy(files[0], mask_recon_dir)

            # run the reconstruction
            reconstruct_volume(
                input_recon_dir,
                mask_recon_dir,
                recon_dir,
                subject_number,
                algorithm,
                HPC,
            )

            # create a json file with the reconstruction parameters
            create_description_file(
                os.path.join(input_dir, "derivatives", algorithm), algorithm
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

    # flag that if set, will indicate we are running this in the HPC cluster
    parser.add_argument(
        "--HPC",
        action="store_true",
        default=False,
        help="If set, will run the reconstruction on the HPC cluster",
    )

    # flag to decide which algorithm to use
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["niftymic", "nesvor", "svrtk"],
        default="niftymic",
        help="Algorithm to use for the reconstruction. Default is niftymic, other option is nesvor",
    )

    # list of str, optional, of the subjects to run the reconstruction on
    # if not set, will run on all subjects
    parser.add_argument(
        "--subject",
        type=str,
        help="Subject to run the reconstruction on. If not set, will run on all subjects. Format: XXX, only the number.",
    )

    # Parameter that forces reconstructions (even if already done)
    parser.add_argument(
        "--force",
        action="store_true",
        help="If set, the reconstruction will be run even if it was already done, removing existing files",
    )

    # Parameter that makes the script only run the preprocessing
    parser.add_argument(
        "--only_preproc",
        action="store_true",
        help="If set, only the preprocessing will be run, not the reconstruction",
    )

    return parser


if __name__ == "__main__":
    parser = parser_obj()
    args = parser.parse_args()

    main(args)
