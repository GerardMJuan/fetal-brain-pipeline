"""Reconstruction functions

"""
import nibabel
import docker
import os
import subprocess
import time
import shutil
import json
from typing import List, Any
import glob 

# GLOBAL VARIABLES
SCRIPT_FILE_NIFTYMIC = "docker_niftymic.sh"
SCRIPT_FILE_NESVOR = "docker_nesvor.sh"

SINGULARITY_NIFTYMIC = "/homedtic/gmarti/SINGULARITY/niftymic_upf_latest.sif"
SINGULARITY_NESVOR = "/homedtic/gmarti/SINGULARITY/nesvor_latest.sif"

DOCKER_NIFTYMIC = "renbem/niftymic"
# TODO: change this to the correct docker
DOCKER_NESVOR = ""



def get_tp(input_recon_dir):
    """
    Gets the through plane resolution of the files in the input_recon_dir
    """
    list_of_files = glob.glob(os.path.join(input_recon_dir, "*.nii.gz"))
    # list of files, remove all that are not nii gz
    list_of_files = [x for x in list_of_files if x.endswith(".nii.gz")]
    tp = 0
    num_scans = 0
    for scan_path in list_of_files:
        # Get voxel dimensions by loading the image with nibabel

        img = nibabel.load(scan_path)
        print(img.shape)

        # get the smallest dimension
        smallest_dim = min(img.shape)
        # get the index
        smallest_dim_index = img.shape.index(smallest_dim)

        # Corresponding pixdim value to extract slice thickness
        pixdim_key = f"pixdim{smallest_dim_index + 1}"

        # Get slice thickness from the identified dimension
        cmd_slice_thickness = ["fslval", scan_path, pixdim_key]
        cmd_slice_thickness = " ".join(cmd_slice_thickness)
        slice_thickness_output = subprocess.run(
            cmd_slice_thickness,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            shell=True,
        )

        # Extract the slice thickness
        slice_thickness = float(
            slice_thickness_output.stdout.decode("utf-8").strip()
        )

        tp += slice_thickness
        num_scans += 1
    avg_tp = tp / num_scans

    return avg_tp


def get_all_thickness(input_recon_dir):
    """
    Gets the through plane resolution of the files in the input_recon_dir

    Not the average, but all of them
    """
    list_of_files = glob.glob(os.path.join(input_recon_dir, "*.nii.gz"))
    # list of files, remove all that are not nii gz
    list_of_files = [x for x in list_of_files if x.endswith(".nii.gz")]

    # sort the files
    list_of_files = sorted(list_of_files)

    tp = []
    for scan_path in list_of_files:
        # Get voxel dimensions by loading the image with nibabel

        img = nibabel.load(scan_path)

        # get the smallest dimension
        smallest_dim = min(img.shape)
        # get the index
        smallest_dim_index = img.shape.index(smallest_dim)

        # Corresponding pixdim value to extract slice thickness
        pixdim_key = f"pixdim{smallest_dim_index + 1}"

        # Get slice thickness from the identified dimension
        cmd_slice_thickness = ["fslval", scan_path, pixdim_key]
        cmd_slice_thickness = " ".join(cmd_slice_thickness)
        slice_thickness_output = subprocess.run(
            cmd_slice_thickness,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            shell=True,
        )

        # Extract the slice thickness
        slice_thickness = float(
            slice_thickness_output.stdout.decode("utf-8").strip()
        )

        tp.append(slice_thickness)

    return tp





def reconstruct_subject(subject, bids_dir, is_HPC, algorithm, stacks, args):
    """Reconstruct the subject.

    Parameters
    ----------
    subject : string
        Subject ID.
    bids_dir : string
        Path to the BIDS dataset.
    is_HPC : bool
        Whether the script is being run on the HPC.
    algorithm : string
        Name of the algorithm to run.
    args : dictionary
        Dictionary containing the arguments passed to the script.
    """

    if is_HPC:
        # call the function depending on the algorithm
        # Now, depending on the algorithm, run the reconstruction
        recon_dir = os.path.join(bids_dir, "recon")
        input_recon_dir = os.path.join(bids_dir, "input")
        mask_dir = os.path.join(bids_dir, "masks")

        # Create the masks if they don't exist
        if not os.path.exists(mask_dir) and os.listdir(mask_dir) == []:
            create_brain_masks(subject, bids_dir)

        # list of files, all files in input_recon_dir that end with nii.gz
        list_of_files = os.listdir(input_recon_dir)
        list_of_files = [x for x in list_of_files if x.endswith(".nii.gz")]

        # list of masks, all files in mask_dir that end with nii.gz
        list_of_masks = os.listdir(mask_dir)
        list_of_masks = [x for x in list_of_masks if x.endswith(".nii.gz")]

        if algorithm == "nesvor":
            reconstruct_nesvor_HPC(
                subject,
                list_of_files,
                list_of_masks,
                bids_dir,
                recon_dir,
                stacks,
                args,
            )
        elif algorithm == "niftymic":
            reconstruct_niftymic_HPC(
                subject,
                list_of_files,
                list_of_masks,
                bids_dir,
                recon_dir,
                stacks,
                args,
            )
        else:
            raise ValueError("Algorithm not recognized.")

    else:
        # call the function depending on the algorithm
        if algorithm == "nesvor":
            reconstruct_nesvor_docker(
                subject,
                list_of_files,
                list_of_masks,
                bids_dir,
                recon_dir,
                stacks,
                args,
            )
        elif algorithm == "niftymic":
            reconstruct_niftymic_docker(
                subject,
                list_of_files,
                list_of_masks,
                bids_dir,
                recon_dir,
                stacks,
                args,
            )
        else:
            raise ValueError("Algorithm not recognized.")


def create_brain_masks(subject, bids_dir):
    """Create brain masks

    Parameters
    ----------
    bids_dir : string
        BIDS directory containing the input data. Should have a folder called input with the input images.
    """

    input_recon_dir = os.path.join(bids_dir, "input")

    # list of files, all files in input_recon_dir that end with nii.gz
    list_of_files = os.listdir(input_recon_dir)
    list_of_files = [x for x in list_of_files if x.endswith(".nii.gz")]

    mask_dir = os.path.join(bids_dir, "masks")
    os.makedirs(mask_dir, exist_ok=True)

    # first run the segmentation if needed
    segmentation_cmd = f"niftymic_segment_fetal_brains --filenames {' '.join(str(x) for x in list_of_files)} --dir-output {mask_dir}"

    # run the command
    with open(
        os.path.join(bids_dir, f"{subject}_mask.log"), "w", encoding="utf8"
    ) as f:
        # first run segmentation
        subprocess.run(
            f"singularity exec --nv {SINGULARITY_NIFTYMIC} {segmentation_cmd}",
            stdout=f,
            stderr=f,
            shell=True,
            check=True,
        )


def bids_format(recon_file, subject, stacks, bids_dir, mask_file=None):
    """Format the output data to BIDS format.

    Parameters
    ----------
    recon_file : string
        Path to the reconstructed file.
    stacks : list of strings
        List of stacks used to reconstruct the image.
    bids_dir : string
        Path to the BIDS dataset.
    mask_file : string, optional
        mask file generated by the recon algorithm, by default None
    """
    bids_path = f"{bids_dir}/sub-{subject}_ses-01_T2w.nii.gz"
    shutil.copy(recon_file, bids_path)

    json_file_path = f"{bids_dir}/sub-{subject}_ses-01_T2w.json"
    data = {"stacks": stacks}
    with open(json_file_path, "w", encoding="utf8") as f:
        json.dump(data, f, indent=4)

    if mask_file is not None:
        bids_path = f"{bids_dir}/sub-{subject}_ses-01_T2w_mask.nii.gz"
        shutil.copy(mask_file, bids_path)


def reconstruct_nesvor_HPC(
    subject: str,
    list_of_files: List[str],
    list_of_masks: List[str],
    bids_dir: str,
    recon_dir: str,
    stacks: Any,
    args: Any,
) -> None:
    """
    Runs the nesvor reconstruction command on the HPC using the specified parameters
    and saves the output in a specified directory. After the reconstruction, it
    checks the success of the operation and logs the information, including the
    reconstruction time. Finally, it formats the output data to BIDS format.

    Parameters
    ----------
    subject : str
        Identifier for the subject to be reconstructed.
    list_of_files : List[str]
        List of file paths to the input stacks to be used in the reconstruction.
    list_of_masks : List[str]
        List of file paths to the stack masks to be used in the reconstruction.
    bids_dir : str
        The directory where the BIDS formatted data will be saved.
    recon_dir : str
        The directory where the reconstruction output volume and logs will be saved.
    stacks : Any
        The stacks used in the reconstruction.
    args : Any
        Additional arguments to be used in the reconstruction, specific type and description depending on your implementation (could be a dict of additional parameters, flags, etc).

    Raises
    ------
    RuntimeError
        Raised when the reconstruction fails, indicating the output file could not be found in the specified reconstruction directory.

    Returns
    -------
    None
        The function does not return a value; it performs operations and saves files to disk.
    """
    print("Reconstructing subject {} using nesvor on the HPC.".format(subject))

    singularity_image = SINGULARITY_NESVOR

    reconstruction_cmd = f"nesvor reconstruct --input-stacks {' '.join(str(x) for x in sorted(list_of_files))} --stack-masks {' '.join(str(x) for x in sorted(list_of_masks))} --output-volume {recon_dir}/nesvor.nii.gz --output-resolution 0.5"

    with open(
        os.path.join(recon_dir, f"{subject}_ses-1_recon.log"),
        "a",
        encoding="utf8",
    ) as f:
        start_time = time.time()

        print(reconstruction_cmd, file=f)
        # second run reconstruction
        subprocess.run(
            f"singularity exec --nv {singularity_image} {reconstruction_cmd}",
            stdout=f,
            stderr=f,
            shell=True,
            check=True,
        )
        end_time = time.time()
        print(f"Reconstruction took {end_time - start_time} seconds", file=f)

    # Check if the reconstruction was successful
    if not os.path.exists(os.path.join(recon_dir, "nesvor.nii.gz")):
        raise RuntimeError("Reconstruction failed.")
    else:
        print(
            f"Reconstruction successful, took {end_time - start_time} seconds"
        )

    result_img = f"{recon_dir}/nesvor.nii.gz"

    # Format the output data to BIDS format
    bids_format(result_img, subject, stacks, bids_dir)


def reconstruct_niftymic_HPC(
    subject, list_of_files, list_of_masks, bids_dir, recon_dir, stacks, args
):
    """
    Conducts a medical image reconstruction using the NiftyMIC pipeline on a High Performance Computing (HPC)
    environment utilizing the Singularity container platform. The function forms a reconstruction command
    incorporating the necessary parameters and executes it within a Singularity container hosting the NiftyMIC software.

    Parameters
    ----------
    subject : str
        The identifier for the individual subject whose data is to be reconstructed.
    list_of_files : List[str]
        List containing the file paths to the images to be used in the reconstruction process.
    list_of_masks : List[str]
        List containing the file paths to the masks to be used in the reconstruction process.
    bids_dir : str
        Directory path where the output data, conforming to BIDS standards, will be stored.
    recon_dir : str
        Directory path where intermediate and final reconstruction outputs will be saved.
    stacks : Any
        Information about the image stacks.
    args : Any

    Raises
    ------
    RuntimeError
        Raised if the reconstruction process fails, determined
        by the absence of the expected output file in the defined directory.

    Returns
    -------
    None
        The function does not return any value;
        it writes outputs to the file system and logs
        the process in a log file.
    """
    print("Reconstructing subject {} using nesvor on the HPC.".format(subject))

    singularity_image = SINGULARITY_NIFTYMIC

    reconstruction_cmd = f"niftymic_run_reconstruction_pipeline --filenames {' '.join(str(x) for x in sorted(list_of_files))} --filenames-masks {' '.join(str(x) for x in sorted(list_of_masks))} --dir-output {recon_dir} --alpha {args.alpha}"

    with open(
        os.path.join(recon_dir, f"{subject}_ses-1_recon.log"),
        "a",
        encoding="utf8",
    ) as f:
        start_time = time.time()

        print(reconstruction_cmd, file=f)
        # second run reconstruction
        subprocess.run(
            f"singularity exec --nv {singularity_image} {reconstruction_cmd}",
            stdout=f,
            stderr=f,
            shell=True,
            check=True,
        )
        end_time = time.time()
        print(f"Reconstruction took {end_time - start_time} seconds", file=f)

    # Check if the reconstruction was successful
    if not os.path.exists(os.path.join(recon_dir, "nesvor.nii.gz")):
        raise RuntimeError("Reconstruction failed.")
    else:
        print(
            f"Reconstruction successful, took {end_time - start_time} seconds"
        )

    result_img = f"{recon_dir}/recon_template_space/srr_template.nii.gz"
    result_mask = f"{recon_dir}/recon_t#emplate_space/srr_template_mask.nii.gz"

    # Format the output data to BIDS format
    bids_format(result_img, subject, stacks, bids_dir, result_mask)


def create_brain_mask_docker(list_of_files, list_of_masks):
    docker_image = DOCKER_NESVOR

    client = docker.from_env()
    with open("recon.log", "a", encoding="utf8") as f:
        container = client.containers.run(
            docker_image,
            user=os.getuid(),
            command=[
                "niftymic_segment_fetal_brains",
                "--filenames",
                list_of_files,
                "--filenames-masks",
                list_of_masks,
            ],
            volumes={
                out_bids_dir: {"bind": "/data", "mode": "rw"},
            },
            detach=True,
            stderr=False,
            stdout=False,
        )
        container.wait()
        # save the logs to f
        f.write(container.logs(stdout=True, stderr=True).decode("utf-8"))
        container.remove()


def reconstruct_nesvor_docker(
    subject: str,
    list_of_files: List[str],
    list_of_masks: List[str],
    bids_dir: str,
    recon_dir: str,
    stacks: Any,
    args: Any,
) -> None:
    """
    Performs a medical image reconstruction using the nesvor pipeline within a Docker container environment. This function constructs a reconstruction command using the supplied parameters, executes this command in Docker, and logs the process in a designated file. After the reconstruction, it checks for success and formats the output data to BIDS standards.

    Parameters
    ----------
    subject : str
        The identifier for the individual subject whose data is to be reconstructed.
    list_of_files : List[str]
        List containing the file paths to the images to be utilized in the reconstruction process.
    list_of_masks : List[str]
        List containing the file paths to the masks to be utilized in the reconstruction process.
    bids_dir : str
        Directory path where the output data, conforming to the BIDS standards, will be stored.
    recon_dir : str
        Directory path where intermediate and final reconstruction outputs will be saved.
    stacks : Any
        Information pertaining to the image stacks.

    Raises
    ------
    RuntimeError
        Raised if the reconstruction process fails.

    Returns
    -------
    None
        The function does not return any value
    """
    print(f"Reconstructing subject {subject} using nesvor in Docker.")

    docker_image = (
        "YOUR_DOCKER_IMAGE_NAME"  # replace with your Docker image name
    )

    reconstruction_cmd = (
        f"nesvor reconstruct --input-stacks {' '.join(map(str, sorted(list_of_files)))} "
        f"--stack-masks {' '.join(map(str, sorted(list_of_masks)))} "
        f"--output-volume /data/nesvor.nii.gz "
        f"--output-resolution 0.5"
    )

    client = docker.from_env()

    with open(
        os.path.join(recon_dir, f"{subject}_ses-1_recon.log"),
        "a",
        encoding="utf8",
    ) as f:
        start_time = time.time()

        print(reconstruction_cmd, file=f)

        response = client.containers.run(
            docker_image,
            command=reconstruction_cmd,
            volumes={recon_dir: {"bind": "/data", "mode": "rw"}},
            remove=True,
        )
        f.write(response.decode())

        end_time = time.time()
        print(f"Reconstruction took {end_time - start_time} seconds", file=f)

    # Check if the reconstruction was successful
    if not os.path.exists(os.path.join(recon_dir, "nesvor.nii.gz")):
        raise RuntimeError("Reconstruction failed.")
    else:
        print(
            f"Reconstruction successful, took {end_time - start_time} seconds"
        )

    result_img = f"{recon_dir}/nesvor.nii.gz"

    # Format the output data to BIDS format
    bids_format(result_img, subject, stacks, bids_dir)


def reconstruct_niftymic_docker(
    subject: str,
    list_of_files: List[str],
    list_of_masks: List[str],
    bids_dir: str,
    recon_dir: str,
    stacks: Any,
) -> None:
    """
    Reconstructs MRI images using the niftymic algorithm via a Docker container.

    Parameters
    ----------
    subject : str
        The identifier of the subject to be reconstructed.
    list_of_files : List[str]
        List of paths to the files to be used in the reconstruction.
    list_of_masks : List[str]
        List of paths to the masks to be used in the reconstruction.
    bids_dir : str
        The directory where the BIDS formatted data is stored.
    recon_dir : str
        The directory where the reconstruction should be saved.
    stacks : Any
        Additional data structures to handle stack related information.
    args : Any
        Additional arguments to control reconstruction parameters such as alpha.
    Raises
    ------
    RuntimeError
        Raised if the reconstruction process fails.
    """

    print(f"Reconstructing subject {subject} using niftymic in Docker.")

    docker_image = DOCKER_NIFTYMIC

    reconstruction_cmd = (
        f"niftymic_run_reconstruction_pipeline --filenames {' '.join(map(str, sorted(list_of_files)))} "
        f"--filenames-masks {' '.join(map(str, sorted(list_of_masks)))} "
        f"--dir-output /data"
    )

    client = docker.from_env()

    with open(
        os.path.join(recon_dir, f"{subject}_ses-1_recon.log"),
        "a",
        encoding="utf8",
    ) as f:
        start_time = time.time()

        print(reconstruction_cmd, file=f)

        response = client.containers.run(
            docker_image,
            command=reconstruction_cmd,
            volumes={recon_dir: {"bind": "/data", "mode": "rw"}},
            remove=True,
        )
        f.write(response.decode())

        end_time = time.time()
        print(f"Reconstruction took {end_time - start_time} seconds", file=f)

    # Check if the reconstruction was successful
    result_img = f"{recon_dir}/recon_template_space/srr_template.nii.gz"
    result_mask = f"{recon_dir}/recon_template_space/srr_template_mask.nii.gz"

    if not os.path.exists(result_img):
        raise RuntimeError("Reconstruction failed.")
    else:
        print(
            f"Reconstruction successful, took {end_time - start_time} seconds"
        )

    # Format the output data to BIDS format
    bids_format(result_img, subject, stacks, bids_dir, result_mask)
