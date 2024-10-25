"""Functions shared by all modules.
"""

import os
import json
import shutil
import time
import subprocess
import docker
import pandas as pd
import nibabel as nib
from nipype.interfaces.ants import DenoiseImage
from nipype import Workflow, Node
import copy
import numpy as np


def apply_masks(list_fo_files, list_of_masks):
    """apply the corresponding mask to each file, overwriting it"""
    for file, mask in zip(list_fo_files, list_of_masks):
        subprocess.call(
            [
                "fslmaths",
                file,
                "-mas",
                mask,
                file,
            ]
        )


def crop_fov(list_of_files, list_of_outputs):
    """
    Crops the fov of the input files using robustfov FSL

    UNUSED
    """
    for base_file, out_file in zip(list_of_files, list_of_outputs):
        subprocess.call(
            [
                "fslreorient2std",
                base_file,
                out_file,
            ]
        )
        subprocess.call(
            [
                "robustfov",
                "-i",
                out_file,
                "-r",
                out_file,
            ]
        )


def denoise_image(in_path, out_path):
    """
    Use ANTs to denoise an image.
    """

    # Run denoise image using a subprocess call
    subprocess.call(
        [
            "DenoiseImage",
            "-i",
            in_path,
            "-o",
            out_path,
            "-n",
            "Gaussian",
            "-s",
            "1",
        ]
    )

def bias_correction(in_path, out_path):
    """
    Use ANTs to perform bias correction on an image.
    """

    # Run N4BiasFieldCorrection using a subprocess call
    subprocess.call(
        [
            "N4BiasFieldCorrection",
            "-d",
            "3",
            "-i",
            in_path,
            "-o",
            out_path,
        ]
    )



def squeeze_dim(arr, dim):
    if arr.shape[dim] == 1 and len(arr.shape) > 3:
        return np.squeeze(arr, axis=dim)
    return arr


def get_cropped_stack_based_on_mask(
    image_ni, mask_ni, boundary_i=15, boundary_j=15, boundary_k=0, unit="mm"
):
    """
    Crops the input image to the field of view given by the bounding box
    around its mask.
    Original code by Michael Ebner:
    https://github.com/gift-surg/NiftyMIC/blob/master/niftymic/base/stack.py

    Input
    -----
    image_ni:
        Nifti image
    mask_ni:
        Corresponding nifti mask
    boundary_i:
    boundary_j:
    boundary_k:
    unit:
        The unit defining the dimension size in nifti

    Output
    ------
    image_cropped:
        Image cropped to the bounding box of mask_ni
    mask_cropped
        Mask cropped to its bounding box
    """

    image_ni = copy.deepcopy(image_ni)

    image = squeeze_dim(image_ni.get_fdata(), -1)
    mask = squeeze_dim(mask_ni.get_fdata(), -1)

    assert all(
        [i >= m] for i, m in zip(image.shape, mask.shape)
    ), "For a correct cropping, the image should be larger or equal to the mask."

    # Get rectangular region surrounding the masked voxels
    [x_range, y_range, z_range] = get_rectangular_masked_region(mask)

    if np.array([x_range, y_range, z_range]).all() is None:
        print("Cropping to bounding box of mask led to an empty image.")
        return None

    if unit == "mm":
        spacing = image_ni.header.get_zooms()
        boundary_i = np.round(boundary_i / float(spacing[0]))
        boundary_j = np.round(boundary_j / float(spacing[1]))
        boundary_k = np.round(boundary_k / float(spacing[2]))

    shape = [min(im, m) for im, m in zip(image.shape, mask.shape)]
    x_range[0] = np.max([0, x_range[0] - boundary_i])
    x_range[1] = np.min([shape[0], x_range[1] + boundary_i])

    y_range[0] = np.max([0, y_range[0] - boundary_j])
    y_range[1] = np.min([shape[1], y_range[1] + boundary_j])

    z_range[0] = np.max([0, z_range[0] - boundary_k])
    z_range[1] = np.min([shape[2], z_range[1] + boundary_k])

    new_origin = list(
        nib.affines.apply_affine(
            mask_ni.affine, [x_range[0], y_range[0], z_range[0]]
        )
    ) + [1]
    new_affine = image_ni.affine
    new_affine[:, -1] = new_origin
    image_cropped = crop_image_to_region(image, x_range, y_range, z_range)
    image_cropped = nib.Nifti1Image(image_cropped, new_affine)
    # image_cropped.header.set_xyzt_units(2)
    # image_cropped.header.set_qform(new_affine, code="aligned")
    # image_cropped.header.set_sform(new_affine, code="scanner")
    return image_cropped


def get_gestational_weeks(subject: str, participants: pd.DataFrame) -> int:
    """
    Retrieve the gestational weeks for a given subject from the participants DataFrame.

    Parameters:
    subject (str): The subject ID.
    participants (pandas.DataFrame): The participants DataFrame.

    Returns:
    int: The gestational weeks for the given subject.
    """
    # add 2 trailing zeros to the subject id as prefix
    subject = subject.zfill(3) if str(subject).isdigit() else subject
    gestational_weeks = participants.loc[
        participants["participant_id"] == f"sub-{subject}", "gestational_weeks"
    ].values[0]
    return gestational_weeks


def get_subjects_and_stacks(args, bids_layout):
    """
    Retrieves the subjects and stacks based on the input arguments.
    Returns lists of subjects and stacks.
    """
    if args.subject is None:
        subjects = bids_layout.get_subjects()
        subjects.sort()
    else:
        subjects = [args.subject]

    return subjects


def create_description_file(recon_dir, algorithm, container="singularity"):
    """Create a dataset_description.json file in the derivatives folder.

    Parameters
    ----------
    args : dictionary
        Dictionary containing the arguments passed to the script.
    container_type : string
        Type of container used to run the algorithm.

    TODO: should look for the extra parameters and also add them to the description file.
    """
    if not os.path.exists(os.path.join(recon_dir, "dataset_description.json")):
        description = {
            "Name": algorithm,
            "Version": "1.0.0",
            "BIDSVersion": "1.7.0",
            "PipelineDescription": {
                "Name": algorithm,
                "Version": "0.6",
            },
            "GeneratedBy": [
                {
                    "Name": algorithm,
                    "Version": "0.1",
                    "Container": {
                        "Type": container,
                        "container": algorithm,
                    },
                }
            ],
        }
        with open(
            os.path.join(
                recon_dir,
                "dataset_description.json",
            ),
            "w",
            encoding="utf-8",
        ) as outfile:
            json.dump(description, outfile)


def remove_directory_contents(path: str):
    """
    Removes all files and directories within the specified path.

    :param path: Path to the directory whose contents are to be removed.
    :raises FileNotFoundError: If the specified path does not exist.
    :raises PermissionError: If there are insufficient permissions to remove the contents.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"The path {path} does not exist.")

    for root, directories, files in os.walk(path, topdown=False):
        for file_name in files:
            os.remove(os.path.join(root, file_name))
        for dir_name in directories:
            os.rmdir(os.path.join(root, dir_name))

    shutil.rmtree(path)


def create_brain_masks(
    list_of_files,
    list_of_masks,
    HPC,
    sing_niftymic_image,
    docker_niftymic,
    gpu=False,
):
    """
    Creates the brain masks using NiftyMIC

    GPU disabled by default
    """
    start_time = time.time()

    if HPC:
        command = [
            "singularity",
            "exec",
            "--nv",
            sing_niftymic_image,
            "niftymic_segment_fetal_brains",
            "--filenames",
            " ".join(str(x) for x in sorted(list_of_files)),
            "--filenames-masks",
            " ".join(str(x) for x in sorted(list_of_masks)),
        ]
        command = " ".join(command)
        # use singularity
        subprocess.call(command, shell=True)
    else:
        # get base directory of the files (it will be the same for all)
        base_dir = os.path.dirname(list_of_files[0])
        # replace the base dir with docker dir /app/NiftyMIC/nifti/
        list_of_files = [
            x.replace(base_dir, "/app/NiftyMIC/nifti") for x in list_of_files
        ]
        base_dir_mask = os.path.dirname(list_of_masks[0])

        list_of_masks = [
            x.replace(base_dir_mask, "/app/NiftyMIC/masks")
            for x in list_of_masks
        ]
        command = [
            "niftymic_segment_fetal_brains",
            "--filenames",
            " ".join(str(x) for x in sorted(list_of_files)),
            "--filenames-masks",
            " ".join(str(x) for x in sorted(list_of_masks)),
        ]
        command = " ".join(command)
        print(f"Running command: {command}")
        client = docker.from_env()

        # check if gpu is available
        if not gpu:
            dev_req = []
        else:
            dev_req = [
                docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
            ]

        container = client.containers.run(
            docker_niftymic,
            user=os.getuid(),
            command=command,
            volumes={
                base_dir: {"bind": "/app/NiftyMIC/nifti", "mode": "rw"},
                base_dir_mask: {"bind": "/app/NiftyMIC/masks", "mode": "rw"},
            },
            device_requests=dev_req,
            detach=True,
            stderr=True,
            stdout=True,
        )
        container.wait()
        # print the logs
        print(container.logs().decode("utf-8"))
        container.remove()

    end_time = time.time()
    print(f"Mask creation took {end_time - start_time} seconds")


def crop_image_to_region(
    image: np.ndarray,
    range_x: np.ndarray,
    range_y: np.ndarray,
    range_z: np.ndarray,
) -> np.ndarray:
    """
    Crop given image to region defined by voxel space ranges
    Original code by Michael Ebner:
    https://github.com/gift-surg/NiftyMIC/blob/master/niftymic/base/stack.py

    Input
    ------
    image: np.array
        image which will be cropped
    range_x: (int, int)
        pair defining x interval in voxel space for image cropping
    range_y: (int, int)
        pair defining y interval in voxel space for image cropping
    range_z: (int, int)
        pair defining z interval in voxel space for image cropping

    Output
    ------
    image_cropped:
        The image cropped to the given x-y-z region.
    """
    image_cropped = image[
        range_x[0] : range_x[1],
        range_y[0] : range_y[1],
        range_z[0] : range_z[1],
    ]
    return image_cropped
    # Return rectangular region surrounding masked region.
    #  \param[in] mask_sitk sitk.Image representing the mask
    #  \return range_x pair defining x interval of mask in voxel space
    #  \return range_y pair defining y interval of mask in voxel space
    #  \return range_z pair defining z interval of mask in voxel space


def get_rectangular_masked_region(
    mask: np.ndarray,
) -> tuple:
    """
    Computes the bounding box around the given mask
    Original code by Michael Ebner:
    https://github.com/gift-surg/NiftyMIC/blob/master/niftymic/base/stack.py

    Input
    -----
    mask: np.ndarray
        Input mask
    range_x:
        pair defining x interval of mask in voxel space
    range_y:
        pair defining y interval of mask in voxel space
    range_z:
        pair defining z interval of mask in voxel space
    """
    if np.sum(abs(mask)) == 0:
        return None, None, None
    shape = mask.shape
    # Compute sum of pixels of each slice along specified directions
    sum_xy = np.sum(mask, axis=(0, 1))  # sum within x-y-plane
    sum_xz = np.sum(mask, axis=(0, 2))  # sum within x-z-plane
    sum_yz = np.sum(mask, axis=(1, 2))  # sum within y-z-plane

    # Find masked regions (non-zero sum!)
    range_x = np.zeros(2)
    range_y = np.zeros(2)
    range_z = np.zeros(2)

    # Non-zero elements of numpy array nda defining x_range
    ran = np.nonzero(sum_yz)[0]
    range_x[0] = np.max([0, ran[0]])
    range_x[1] = np.min([shape[0], ran[-1] + 1])

    # Non-zero elements of numpy array nda defining y_range
    ran = np.nonzero(sum_xz)[0]
    range_y[0] = np.max([0, ran[0]])
    range_y[1] = np.min([shape[1], ran[-1] + 1])

    # Non-zero elements of numpy array nda defining z_range
    ran = np.nonzero(sum_xy)[0]
    range_z[0] = np.max([0, ran[0]])
    range_z[1] = np.min([shape[2], ran[-1] + 1])

    # Numpy reads the array as z,y,x coordinates! So swap them accordingly
    return (
        range_x.astype(int),
        range_y.astype(int),
        range_z.astype(int),
    )
