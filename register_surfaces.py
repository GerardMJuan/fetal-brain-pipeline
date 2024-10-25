"""
Script that registers the surface obtained by the dhcp pipeline to the 
fetal atlas by using the aMSM algorithm.

It needs to download the fetal atlas and the MSM program from the following links:
aMSM algorithm: https://www.doc.ic.ac.uk/~ecr05/MSM_HOCR_v2
fetal atlas: https://gin.g-node.org/kcl_cdb/dhcp_fetal_brain_surface_atlas

workbench needs to be installed too.

And indicate the paths in the config file.

Uses the same format as the other scripts.

Needs a BIDS-compliant folder as input.

Run example:
python register_surfaces.py --input CMV_BIDS --i_derivative niftymic-dhcp --output niftymic-dhcp-surfreg --subject 008
"""

import os
import argparse
import subprocess
import pandas as pd
import bids as pybids
import json
import docker
import sys
import glob
import numpy as np
from source.utils import get_gestational_weeks, create_description_file
from source.freesurfer import dhcp_to_freesurfer
import time


def get_atlas_path(gestational_weeks, paths):
    """
    Returns the path to the atlas for the given gestational weeks.
    """
    atlas_base_path = paths["atlas_path"]

    # convert ga to the closest week (integer) and convert to string
    ga = str(int(np.round(gestational_weeks)))
    if int(ga) > 36:
        ga = "36"

    atlas_L = f"{atlas_base_path}/atlas/fetal.week{ga}.left.sphere.surf.gii"
    atlas_R = f"{atlas_base_path}/atlas/fetal.week{ga}.right.sphere.surf.gii"

    sulc_L = f"{atlas_base_path}/atlas/fetal.week{ga}.left.sulc.shape.gii"
    sulc_R = f"{atlas_base_path}/atlas/fetal.week{ga}.right.sulc.shape.gii"

    midthickness_L = (
        f"{atlas_base_path}/atlas/fetal.week{ga}.left.midthickness.surf.gii"
    )
    midthickness_R = (
        f"{atlas_base_path}/atlas/fetal.week{ga}.right.midthickness.surf.gii"
    )

    return {
        "L": {
            "atlas": atlas_L,
            "sulc": sulc_L,
            "midthickness": midthickness_L,
        },
        "R": {
            "atlas": atlas_R,
            "sulc": sulc_R,
            "midthickness": midthickness_R,
        },
    }


def main(args):
    try:
        layout = pybids.BIDSLayout(args.input, derivatives=True)
    except AssertionError:
        print("The input directory is not a BIDS directory, exiting...")
        sys.exit()

    # Specify the JSON file name
    json_file_path = args.config

    # Load the JSON file
    with open(json_file_path, "r", encoding="utf8") as json_file:
        paths = json.load(json_file)

    # load the participants.tsv file
    participants = pd.read_csv(
        os.path.join(args.input, "participants.tsv"), sep="\t"
    )

    derivative_output_dir = os.path.join(
        args.input, "derivatives", args.output
    )
    # Create output directory as new derivative folder
    if not os.path.exists(derivative_output_dir):
        os.makedirs(derivative_output_dir)

    create_description_file(derivative_output_dir, args.output)

    # get subject if specified, else, get all subjects
    if args.subject is not None:
        subjects = [args.subject]

        # check if the gestational weeks is specified
        if args.ga is not None:
            gestational_weeks = args.ga
        else:
            gestational_weeks = get_gestational_weeks(
                args.subject, participants
            )
    else:
        subjects = layout.get_subjects()
        subjects.sort()
        gestational_weeks = None

    # for subject in subjects
    for subject in subjects:
        # Check if that subject surface has been reconstructed
        # in the args.i_derivative folder
        # by looking at the surface/workbench/ folder and check if files exist
        # if not, skip that subject
        sub = f"sub-{subject}"
        ses = "ses-01"
        suffix = "T2w"

        # surface folder
        surface_dir = os.path.join(
            args.input,
            "derivatives",
            args.i_derivative,
            sub,
            ses,
            "anat",
            "surfaces",
            f"{sub}_{ses}_{suffix}",
            "workbench",
        )

        input_dhcp = os.path.join(
            args.input,
            "derivatives",
            args.i_derivative,
            sub,
            ses,
            "anat",
        )
        output_folder = os.path.join(
            args.input,
            "derivatives",
            args.i_derivative,
            sub,
            ses,
            "anat",
            "freesurfer",
        )
        # create output folder if it does not exist
        os.makedirs(output_folder, exist_ok=True)

        # Check if file *.sphere.native.surf.gii exists
        # if not, skip that subject
        if not os.path.exists(
            os.path.join(
                surface_dir,
                f"{sub}_{ses}_{suffix}.L.sphere.native.surf.gii",
            )
        ):
            print(
                f"Subject {subject} does not have a surface in {args.i_derivative} folder, skipping..."
            )
            continue

        # get the subject's folder
        output_dir = os.path.join(
            derivative_output_dir, f"sub-{subject}", ses, "anat/"
        )
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # get gestational weeks only if not specified
        if gestational_weeks is None:
            gestational_weeks = get_gestational_weeks(subject, participants)

        if gestational_weeks is None:
            print("Gestational weeks not specified for subject: " + subject)
            continue

        # get the correct atlas path
        # atlas path is a dictionary with only two keys: "L" and "R". This, in turn,
        # contains a dictiionarity with the various atlas paths and data for that
        # gestational week and hemisphere
        atlas_dict = get_atlas_path(gestational_weeks, paths)
        ### START OF REGISTRATION ###

        # do it for each hemisphere
        for h in ["L", "R"]:
            # if the output file already exists, skip

            # registration script path
            msm_path = paths["msm_path"]
            msm_config = paths["msmall_config_path"]
            atlas_path = atlas_dict[h]["atlas"]
            atlas_sulc = atlas_dict[h]["sulc"]
            input_path = os.path.join(
                surface_dir,
                f"{sub}_{ses}_{suffix}.{h}.sphere.native.surf.gii",
            )
            input_sulc = os.path.join(
                surface_dir,
                f"{sub}_{ses}_{suffix}.{h}.sulc.native.shape.gii",
            )
            out_name = f"{output_dir}{sub}_{ses}_{suffix}.{h}."

            
            if os.path.exists(
                os.path.join(
                    output_dir,
                    out_name,
                )
            ):
                print(
                    f"Subject {subject} already has a registered surface, skipping..."
                )
                continue

            # we use MSMSulc
            # run the registration scripts
            try:
                print("Running MSM registration...")
                cmd = [
                    f"{msm_path}",
                    f"--conf={msm_config}",
                    f"--inmesh={input_path}",
                    f"--refmesh={atlas_path}",
                    f"--indata={input_sulc}",
                    f"--refdata={atlas_sulc}",
                    f"--out={out_name}",
                ]
                print(" ".join(cmd))

                start_time = time.time()
                result = subprocess.run(
                    cmd,
                    check=True,
                )
                end_time = time.time()
                execution_time = end_time - start_time

                print(
                    f"The subprocess took {execution_time:.2f} seconds to complete."
                )
            except subprocess.CalledProcessError as error:
                print(f"Subprocess failed with error code {error.returncode}")
                print("STDOUT:", error.stdout if error.stdout else "No STDOUT")
                print("STDERR:", error.stderr if error.stderr else "No STDERR")
            except Exception as error:
                print(f"An unexpected error occurred: {error}")
        ### END OF REGISTRATION ###

        # use the obtained registration to register the surfaces
        # to the atlas
        for h in ["L", "R"]:
            # do it using https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/MSM/UserGuide
            # and https://github.com/ecr05/dHCP_template_alignment/blob/master/surface_to_template_alignment/align_to_template_3rd_release.sh (i el og)

            # We assume that the Connectome workbench is installed

            # prepare paths
            reg_sphere = os.path.join(
                output_dir, f"{sub}_{ses}_{suffix}.{h}.sphere.reg.surf.gii"
            )
            atlas_sphere = atlas_dict[h]["atlas"]
            atlas_midthickness = atlas_dict[h]["midthickness"]
            midthickness_surf = os.path.join(
                surface_dir,
                f"{sub}_{ses}_{suffix}.{h}.midthickness.native.surf.gii",
            )

            surface_list = [
                "pial",
                "white",
                "midthickness",
                "inflated",
                "very_inflated",
            ]

            for surf in surface_list:
                surf_path = os.path.join(
                    surface_dir,
                    f"{sub}_{ses}_{suffix}.{h}.{surf}.native.surf.gii",
                )

                output_file = os.path.join(
                    output_dir,
                    f"{sub}_{ses}_{suffix}.{h}.{surf}.atlas.surf.gii",
                )

                cmd = [
                    "wb_command",
                    "-surface-resample",
                    surf_path,  # the surface file to resample
                    reg_sphere,  # the registered sphere
                    atlas_sphere,  # the atlas sphere
                    "ADAP_BARY_AREA",  # resampling method
                    output_file,  # this will be an output file, related to fs?
                    "-area-surfs",
                    midthickness_surf,
                    atlas_midthickness,
                ]

                # run the line
                print(" ".join(cmd))
                result = subprocess.run(
                    cmd,
                    check=True,
                )

            metric_list = [
                "sulc",
                "curvature",
                "thickness",
                "roi",
                "corrThickness",
            ]

            for metric in metric_list:
                metric_path = os.path.join(
                    surface_dir,
                    f"{sub}_{ses}_{suffix}.{h}.{metric}.native.shape.gii",
                )

                output_file = os.path.join(
                    output_dir,
                    f"{sub}_{ses}_{suffix}.{h}.{metric}.atlas.shape.gii",
                )

                cmd = [
                    "wb_command",
                    "-metric-resample",
                    metric_path,  # the metric file to resample
                    reg_sphere,  # the registered sphere
                    atlas_sphere,  # the atlas sphere
                    "ADAP_BARY_AREA",  # resampling method
                    output_file,  # this will be an output file
                    "-area-surfs",
                    midthickness_surf,
                    atlas_midthickness,
                ]

                # run the line
                print(" ".join(cmd))
                result = subprocess.run(
                    cmd,
                    check=True,
                )

            # finally, do the same for drawem
            drawem_path = os.path.join(
                surface_dir,
                f"{sub}_{ses}_{suffix}.{h}.drawem.native.label.gii",
            )
            
            output_file = os.path.join(
                output_dir,
                f"{sub}_{ses}_{suffix}.{h}.drawem.atlas.label.gii",
            )

            cmd = [
                "wb_command",
                "-label-resample",
                drawem_path,  # the label file to resample
                reg_sphere,  # the registered sphere
                atlas_sphere,  # the atlas sphere
                "ADAP_BARY_AREA",  # resampling method
                output_file,  # this will be an output file
                "-area-surfs",
                midthickness_surf,
                atlas_midthickness,
            ]
            # run the line
            print(" ".join(cmd))
            result = subprocess.run(
                cmd,
                check=True,
            )


if __name__ == "__main__":
    # Parse input arguments
    parser = argparse.ArgumentParser(
        description="Script to register surface data to a fetal atlas."
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
    parser.add_argument(
        "--subject",
        type=str,
        help="subject ID (if not specified, all subjects will be processed)",
        required=False,
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
        "--config",
        type=str,
        help="config file, for paths of scripts and atlases",
        required=True,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="if specified, the script will overwrite any existing segmentation",
        required=False,
    )

    args = parser.parse_args()

    main(args)


## Example
# python register_surfaces.py --input ERANEU_MULTIFACT --i_derivative dhcp-nesvor --output dhcp-nesvor-surfreg --subject 295 --config configs/config_hpc_strain.json
#
