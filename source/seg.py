"""
Auxiliary functions for segmentation
"""
import os
import shutil
import subprocess
import glob
from source.paths import SINGULARITY_twai, SINGULARITY_bounti, env

env = {
    **os.environ,
}


def copy_brain_mask(
    out_bids_dir: str, recon_file_name: str, mask_file: str
) -> None:
    """
    Copy the brain mask file to the segmentations directory if it doesn't already exist.

    Parameters:
    out_bids_dir (str): The output BIDS directory.
    recon_file_name (str): The name of the reconstructed T2w file.
    mask_file (str): The path to the brain mask file.
    """
    if not os.path.exists(os.path.join(out_bids_dir, "segmentations")):
        os.makedirs(os.path.join(out_bids_dir, "segmentations"))

    if not os.path.exists(
        os.path.join(
            out_bids_dir,
            "segmentations",
            f"{recon_file_name}_brain_mask.nii.gz",
        )
    ):
        shutil.copyfile(
            mask_file,
            os.path.join(
                out_bids_dir,
                "segmentations",
                f"{recon_file_name}_brain_mask.nii.gz",
            ),
        )


def subject_selection(input_dir, i_derivative, subject):
    """
    Return the input T2w subject, the input brain mask file, and the output BIDS directory.
    """
    if i_derivative != "raw":
        base_path = os.path.join(
            input_dir,
            "derivatives",
            i_derivative,
            f"sub-{subject}",
            "ses-01",
            "anat",
        )
        T2w_file = os.path.join(
            base_path,
            f"sub-{subject}" + "_ses-01_T2w.nii.gz",
        )

        mask_file = os.path.join(
            base_path,
            f"sub-{subject}" + "_ses-01_mask.nii.gz",
        )

    else:
        # could be any file ending with _T2w.nii.gz, find it
        T2w_file = glob.glob(
            os.path.join(input_dir, f"sub-{subject}", "anat", "*_T2w.nii.gz")
        )[0]

        # mask file is the same as T2w_file but with _mask.nii.gz
        mask_file = T2w_file.replace("_T2w.nii.gz", "_mask.nii.gz")

    return T2w_file, mask_file


def check_if_done(out_bids_dir, alg):
    """
    Return True if the segmentation is already done, False otherwise.

    Check the output file depending on the algorithm.
    """
    if alg == "bounti":
        return (
            len(glob.glob(os.path.join(out_bids_dir, "*_bounti-19.nii.gz")))
            > 0
        )
    elif alg == "twai":
        return os.path.exists(
            os.path.join(out_bids_dir, "trustworthyAI", "anat.nii.gz")
        )
    elif alg == "dhcp":
        # return both the surfaces and the segmentations
        return (
            len(
                glob.glob(
                    os.path.join(
                        out_bids_dir, "segmentations", "*_all_labels.nii.gz"
                    )
                )
            )
            > 0
        ) & (
            len(
                glob.glob(
                    os.path.join(
                        out_bids_dir,
                        "surfaces/*/workbench",
                        "*_T2w.L.pial.native.surf.gii",
                    )
                )
            )
            > 0
        )

    else:
        return False


def compute_brain_mask(
    input_dir, subject, T2w_file, out_mask_file, alg, i_derivative
):
    """
    Compute the brain mask.

    Three types of brain masks are available:
    - binarize the T2w image
    - Use BET
    - Use BOUNTI

    If using "raw" (FETA data), do the binarization of the available segmentation.

    Return the path to the brain mask file.
    """

    # Check if mask file eixsts, if it does, remove it and recompute it
    if os.path.exists(out_mask_file):
        os.remove(out_mask_file)

    if alg == "bin":
        to_segment = T2w_file
        if i_derivative == "raw":
            # Binarize the available segmentation
            to_segment = os.path.join(
                input_dir,
                f"sub-{subject}",
                "anat",
                f"sub-{subject}" + "_*_dseg.nii.gz",
            )

            to_segment = glob.glob(to_segment)[0]

        subprocess.run(
            " ".join(
                [
                    "fslmaths",
                    to_segment,
                    "-bin",
                    out_mask_file,
                ]
            ),
            shell=True,
            # env=env,
            check=True,
        )

    elif alg == "bet":
        subprocess.run(
            " ".join(
                [
                    "bet",
                    T2w_file,
                    out_mask_file.split(".")[0],
                    "-R",
                    "-f",
                    "0.15",
                    "-m",
                ]
            ),
            shell=True,
            env=env,
            check=True,
        )

    elif alg == "bounti":
        # Run the BOUNTI algorithm.
        # Using the singularity container.

        # chekc if the mask exists in the derivatives folder
        if i_derivative != "raw":
            base_path = os.path.join(
                input_dir,
                "derivatives",
                "nesvor-bounti",
                f"sub-{subject}",
                "ses-01",
                "anat",
            )
            mask_file = os.path.join(
                base_path,
                f"sub-{subject}" + "_ses-01_T2w-mask-bet-1.nii.gz",
            )
            if os.path.exists(mask_file):
                shutil.copyfile(mask_file, out_mask_file)
                return out_mask_file
            else:
                print("Running BOUNTI algorithm")


        recon_dir = os.path.dirname(T2w_file)

        if not os.path.exists(os.path.join(recon_dir, "input_rec")):
            # rename the input folder inside recon-dir to input_rec
            os.rename(os.path.join(recon_dir, "input"), os.path.join(recon_dir, "input_rec"))

        # Remove input folder
        shutil.rmtree(os.path.join(recon_dir, "input"), ignore_errors=True)

        # create input folder
        os.makedirs(os.path.join(recon_dir, "input"), exist_ok=True)


        # copy the T2w_file to the recon_dir/input folder
        # should have the same name as the original file
        shutil.copyfile(T2w_file, os.path.join(recon_dir, "input", os.path.basename(T2w_file)))
        
        os.makedirs(os.path.join(recon_dir, "tmp_proc"), exist_ok=True)
        pre_command = f"singularity exec -u --nv --bind /home/gmarti/mock_home:/home/gmarti --bind {recon_dir}:/home/data/ --bind {recon_dir}/tmp_proc:/home/tmp_proc"
        image = SINGULARITY_bounti
        command = (
            "bash /home/auto-proc-svrtk/auto-brain-bet-segmentation-fetal.sh"
        )
        args = ["/home/data/input", "/home/data"]

        # Run the command
        subprocess.run(
            " ".join([pre_command, image, command] + args),
            shell=True,
            env=env,
            check=True,
        )

        # Copy the brain mask file to the correct filename
        shutil.copyfile(
            os.path.join(
                recon_dir, f"sub-{subject}" + "_ses-01_T2w-mask-bet-1.nii.gz"
            ),
            out_mask_file,
        )

    return out_mask_file
