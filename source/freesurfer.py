from nibabel import freesurfer as fs
import nibabel as nib
import numpy as np
import os
import subprocess


def fix_gifti_ras(gifti_file, output_folder):
    mesh = nib.load(gifti_file)
    coordinates = mesh.darrays[0].data
    translation = np.array(
        [
            float(mesh.darrays[0].meta["VolGeomC_R"]),
            float(mesh.darrays[0].meta["VolGeomC_A"]),
            float(mesh.darrays[0].meta["VolGeomC_S"]),
        ]
    )

    # Compute updated coordinates
    updated_coordinates = coordinates + translation

    # Compute updated header: keep only non-RAS information
    meta = mesh.darrays[0].meta
    meta_items_to_keep = [
        "AnatomicalStructurePrimary",
        "AnatomicalStructureSecondary",
        "GeometricType",
        "Name",
    ]
    updated_meta = nib.gifti.GiftiMetaData(
        {k: meta[k] for k in meta_items_to_keep}
    )

    # Update gifti file
    mesh.darrays[0].data = updated_coordinates
    mesh.darrays[0].meta = updated_meta

    # Save updated mesh
    mesh_basename = os.path.basename(gifti_file)
    mesh.to_filename(output_folder / mesh_basename)


def dhcp_to_freesurfer(subj, input_dhcp, output_folder):
    """
    Function that converts the output of the dhcp pipeline to freesurfer format

    Parameters
    ----------
    input_dhcp : str
        Path to the output of the dhcp pipeline
    output_folder : str
        Path to the output folder

    Returns nothing
    -------
    """

    # If empty, skip
    if not os.listdir(input_dhcp):
        print("Cannot convert to fs, something went wrong!")
        raise FileNotFoundError

    subj_name = f"sub-{subj}_ses-01_T2w"

    freesurfer_folder = os.path.join(output_folder, subj_name, "surf")
    base_freesurfer_folder = os.path.join(output_folder, subj_name)
    os.makedirs(freesurfer_folder, exist_ok=True)

    for h in [1, 2]:
        hemi, Hemi = ("r", "R") if h == 1 else ("l", "L")

        # TODO: use reg instead of native?
        # define pairs of input and output paths
        i_o_paths = [
            ("pial.native.surf.gii", "h.pial"),
            ("white.native.surf.gii", "h.white"),
            ("inflated.native.surf.gii", "h.inflated"),
            ("sphere.native.surf.gii", "h.sphere"),
            ("sulc.native.shape.gii", "h.sulc"),
            ("curvature.native.shape.gii", "h.curv"),
            ("thickness.native.shape.gii", "h.thickness"),
            ("drawem.native.label.gii", "h.labels"),
        ]

        # Convert N4/T2w image to freesurfer mgz
        subprocess.run(
            [
                "mri_convert",
                os.path.join(input_dhcp, "N4", f"{subj_name}.nii.gz"),
                os.path.join(base_freesurfer_folder, "mri.mgz"),
            ],
            check=True,
        )

        for i_suffix, o_suffix in i_o_paths:
            # Define your input and output paths
            # You need to adapt these to your folder structure
            i_full = os.path.join(
                input_dhcp,
                "surfaces",
                subj_name,
                "workbench",
                f"{subj_name}.{Hemi}.{i_suffix}",
            )

            # i_full_fixed = os.path.join(
            #     input_dhcp,
            #     "surfaces",
            #     subj_name,
            #     "workbench_fixed",
            # )
            # os.makedirs(i_full_fixed, exist_ok=True)
            # # Fix RAS
            # fix_gifti_ras(i_full, i_full_fixed)

            # i_full = os.path.join(
            #     i_full_fixed,
            #     f"{subj_name}.{Hemi}.{i_suffix}",
            # )

            # save pial for later
            if i_suffix == "pial.native.surf.gii":
                ipial = i_full

            o_full = os.path.join(freesurfer_folder, f"{hemi}{o_suffix}")

            if "surf.gii" in i_suffix:
                # Convert all
                subprocess.run(
                    [
                        "mris_convert",
                        i_full,
                        o_full,
                    ],
                    check=True,
                )

                # save_geometry(i_full, o_full)

            # Run your custom Python script for additional conversions
            # only for shape and label
            if "surf.gii" not in i_suffix:
                convert_to_freesurfer(ipial, i_full, o_full)

        # Convert white matter mask to freesurfer mgz
        subprocess.run(
            [
                "fscalc",
                os.path.join(
                    input_dhcp,
                    "segmentations",
                    f"{subj_name}_L_white.nii.gz",
                ),
                "sum",
                os.path.join(
                    input_dhcp,
                    "segmentations",
                    f"{subj_name}_R_white.nii.gz",
                ),
                "-o",
                os.path.join(base_freesurfer_folder, "wm.nii.gz"),
            ],
            check=True,
        )
        subprocess.run(
            [
                "mri_convert",
                os.path.join(base_freesurfer_folder, "wm.nii.gz"),
                os.path.join(base_freesurfer_folder, "wm.mgz"),
            ],
            check=True,
        )


def convert_to_freesurfer(ipial, ipath, opath):
    shape = nib.load(ipial)
    faces = np.array(shape.agg_data("NIFTI_INTENT_TRIANGLE"))
    tri = len(faces)

    shape = nib.load(ipath)
    coords = np.array(shape.agg_data())
    fs.write_morph_data(opath, coords, tri)


def save_geometry(path_in, path_out):
    shape = nib.load(path_in)
    # get attributes of the shape

    print(shape.agg_data)

    vertices = np.array(shape.agg_data("NIFTI_INTENT_POINTSET"))
    faces = np.array(shape.agg_data("NIFTI_INTENT_TRIANGLE"))
    fs.write_geometry(path_out, vertices, faces)
