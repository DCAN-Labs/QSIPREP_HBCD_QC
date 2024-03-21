#!/usr/local/bin/python3
import warnings

import argparse
import glob
import os
import os.path as op
import nibabel as nb
import bids
import imageio
import numpy as np
import subprocess
from PIL import Image

warnings.simplefilter(action="ignore", category=FutureWarning)

# Settings for the gifs
# Specify the slices that we would like to save as a fraction
# of the masked extent.
slice_ratios = np.array([0.30, 0.40, 0.50, 0.60])
slice_axis = 2  # We're taking the z axis
# Specify that slice range for the animated gifs
slice_gif_offsets = np.arange(-5, 6)
# Calculate the frames-per-second for the animated gifs
fps = len(slice_gif_offsets) / 2.0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "qsiprep_dir",
        help="The path to the BIDS directory for your study (this is the same for all subjects)",
        type=str,
    )
    parser.add_argument(
        "output_dir",
        help="The path to the folder where outputs will be stored (this is the same for all subjects)",
        type=str,
    )
    parser.add_argument("analysis_level", help="Should always be participant", type=str)

    parser.add_argument(
        "--participant_label",
        "--participant-label",
        help="The name/label of the subject to be processed (i.e. sub-01 or 01)",
        type=str,
    )
    parser.add_argument(
        "--session_id",
        "--session-id",
        help="OPTIONAL: the name of a specific session to be processed (i.e. ses-01)",
        type=str,
    )
    parser.add_argument(
        "--bids_directory",
        "--bids_directory",
        help="OPTIONAL: This is not actually used for processing.",
        type=str,
    )

    return parser.parse_args()


def create_png_space(hires_maskfile, hires_anatfile):
    """Create an output space that is ideal for creating square png images.

    Parameters
    ----------

    hires_maskfile: str
        Path to the brain mask from qsiprep

    hires_anatfile: str
        Path to high-res anatomical image from qsiprep. Can be T1w or T2w


    Returns
    -------

    pngres_maskfile: str
        Path to the brain mask in png space

    pngres_anatfile: str
        Path to the anatomical image in png space

    """

    # Create a cube bounding box that we will use to take pics
    pngres_anat = op.abspath("pngres_anatomical.nii")
    pngres_mask = op.abspath("pngres_mask.nii")

    subprocess.run(
        ["3dAutobox", "-prefix", "_mask_box.nii", hires_maskfile],
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            "3dZeropad",
            "-RL",
            "128",
            "-AP",
            "128",
            "-IS",
            "128",
            "-prefix",
            pngres_mask,
            "_mask_box.nii",
        ],
        capture_output=True,
        text=True,
    )

    # Get the brainmask in pngref space
    subprocess.run(
        [
            "antsApplyTransforms",
            "-d",
            "3",
            "-i",
            hires_anatfile,
            "-r",
            "pngref.nii",
            "-o",
            pngres_anat,
            "-v",
            "1",
            "--interpolation",
            "NearestNeighbor",
        ]
    )

    # Clean up
    os.remove("_mask_box.nii")

    return pngres_mask, pngres_anat


def dec_in_pngspace(pngres_mask, zipped_preproc_dwi_file):
    """Fit a tensor in TORTOISE and create a DEC image in png space.

    Parameters
    ----------

    pngres_mask: str
        Path to the binary brain mask file in png space

    zipped_preproc_dwi_file: str
        Path to the gzipped, preprocessed nifti file from qsiprep


    Returns
    -------

    pngres_dec: str
        Path to the RGB-valued DEC image in png space
    """
    pngres_dec = op.abspath("pngres_dec.nii")
    # Unzip the dwi file
    subprocess.run(["gunzip", zipped_preproc_dwi_file])
    preproc_dwi_file = op.abspath(zipped_preproc_dwi_file.replace(".gz", ""))
    fstem = preproc_dwi_file.replace(".nii", "")

    # Convert bval, bvec to bmat
    subprocess.run(["FSLBVecsToTORTOISEBmatrix", f"{fstem}.bval", f"{fstem}.bvec"])

    # Estimate the tensor (the bmtxt is automatically found)
    subprocess.run(
        [
            "EstimateTensor",
            "--input",
            preproc_dwi_file,
            "--mask",
            pngres_mask,
            "--bval_cutoff",
            "2200",
        ]
    )
    # Calculate FA
    subprocess.run(["ComputeFAMap", f"{fstem}_L1_DT.nii"])

    # Make the DEC Map
    subprocess.run(["ComputeDECMap", "--input_tensor", f"{fstem}_L1_DT.nii", "--useFA"])

    # Resample the DEC Map into pngres space
    subprocess.run(
        [
            "antsApplyTransforms",
            "-d",
            "3",
            "-e",
            "4",
            "--interpolation",
            "NearestNeighbor",
            "-o",
            pngres_dec,
            "-i",
            f"{fstem}_L1_DT_DEC.nii" "-r",
            pngres_mask,
        ]
    )

    # Clean up
    os.remove(f"{fstem}_L1_DT_DEC.nii")
    os.remove(f"{fstem}_L1_DT.nii")
    os.remove(f"{fstem}_L1_DT_FA.nii")
    os.remove(f"{fstem}_L1_AM.nii")
    os.remove(preproc_dwi_file)  # This is the unzipped nii, the original stays

    return pngres_dec


def create_gifs(bids_dir, subject, output_dir, session=None):
    """Create DEC+anat animated gifs from QSIPrep outputs.

    Parameters
    ----------

    bids_dir: str
        Path to QSIPrep outputs (A BIDS derivatives dataset)

    subject: str
        Subject ID to process

    output_dir: str
        Directory where gifs will go, in a BIDSlike layout

    session: str
        Filter for session. If omitted create gifs from all sessions

    Returns
    -------

    None

    """
    # Download that particular subject to a local folder
    layout = bids.BIDSLayout(bids_dir, validate=False)

    # Specify local filenames
    subject = subject.replace("sub-", "")

    # Use pybids to grab the necessary image files
    initial_bids_filters = {"subject": subject, "return_type": "filename"}

    # Prepare the PNG space and get the anatomical data resampled into it
    anat_mask_file = layout.get(
        suffix="mask",
        datatype="anat",
        extension="nii.gz",
        **initial_bids_filters,
    )
    if not anat_mask_file:
        raise Exception(f"No anatomical brain mask produced for {subject}")
    anat_mask_file = anat_mask_file[0]
    # We get a T2w for HBCD
    anat_hires_file = layout.get(
        suffix="T2w",
        datatype="anat",
        extension="nii.gz",
        **initial_bids_filters,
    )[0]
    if not anat_hires_file:
        raise Exception(f"No high-res T2w image found for f{subject}")
    anat_hires_file = anat_hires_file[0]
    pngres_mask, pngres_anat = create_png_space(anat_mask_file, anat_hires_file)

    # Create the local output dir
    if session is not None:
        session_name = f"ses-{session}"
    else:
        session_name = ""
    png_dir = op.join(
        output_dir,
        f"sub-{subject}/{session_name}/dwi"
    ).replace("//", "/")
    os.makedirs(png_dir, exist_ok=True)

    # The anatomical outputs from QSIPrep v<1.0
    # cannot be in session directories
    # Add this filter after we've found the anatomical data
    if session is not None:
        initial_bids_filters["session"] = session

    # Find all the dwi files, and their corresponding dwi files
    dwi_files = layout.get(suffix="dwi", extension="nii.gz", **initial_bids_filters)
    for dwi_file in enumerate(dwi_files):
        pngres_dwi = dec_in_pngspace(pngres_mask, dwi_file)
        gif_prefix = op.join(
            png_dir,
            f"sub-{subject}_{session_name}_qcgif-".replace("__", "_")
        )
        gifs_from_dec(pngres_dwi, pngres_mask, pngres_anat, prefix=gif_prefix)


def get_anchor_slices_from_mask(mask_file, axis=2):
    """Find the slice numbers for ``slice_ratios`` inside a mask."""
    mask_arr = nb.load(mask_file).get_fdata()
    mask_coords = np.argwhere(mask_arr > 0)[axis]
    min_slice = mask_coords.min()
    max_slice = mask_coords.max()
    covered_slices = max_slice - min_slice
    return np.floor(covered_slices * slice_ratios + min_slice)



def gifs_from_dec(dec_file, mask_file, anat_file, prefix):
    """Create a series of animated gifs from DEC+anatomical images.


    """

    # Compute the indices of the slices
    slice_indices = get_anchor_slices_from_mask(mask_file)

    # Loop is for individual slices in the gif image
    for gif_idx, base_slice_idx in enumerate(slice_indices):
        images = []
        for offset_idx, slice_offset in enumerate(slice_gif_offsets):
            slice_idx = base_slice_idx + slice_offset

            file_path = op.join(
                , fname_gif + str(gif_idx) + "_" + str(offset_idx) + ".png"
            )

            images.append(imageio.imread(file_path))

        images = images + images[-2:0:-1]

        file_path = op.join(png_dir, fname_gif + str(gif_idx) + ".gif")

        # Save the gif
        imageio.mimsave(
            file_path, images, loop=0, duration=(1 / fps) * 1000, subrectangles=True
        )


# Grab the arg parse inputs
args = parse_args()
cwd = os.getcwd()

# reassign variables to command line input
qsiprep_dir = args.qsiprep_dir
if not os.path.isabs(qsiprep_dir):
    qsiprep_dir = os.path.join(cwd, qsiprep_dir)
output_dir = args.output_dir
if not os.path.isabs(output_dir):
    output_dir = os.path.join(cwd, output_dir)
analysis_level = args.analysis_level
if analysis_level != "participant":
    raise ValueError(
        "Error: analysis level must be participant, but program received: "
        + analysis_level
    )

if args.participant_label:
    pass
else:
    os.chdir(qsiprep_dir)
    participants = glob.glob("sub-*")

for temp_participant in participants:

    # Find sessions... if session was provided then
    # process that specific session. Otherwise iterate
    # through all sessions or continue without assumption
    # of sessions if no sessions are found in the BIDS
    # structure

    # This is mainly to weed out html reports...
    dwi_path_sessions = os.path.join(qsiprep_dir, temp_participant, "ses*", "dwi")
    dwi_path_no_sessions = os.path.join(qsiprep_dir, temp_participant, "ses*")
    if len(glob.glob(dwi_path_sessions)) + len(glob.glob(dwi_path_no_sessions)) == 0:
        continue

    if args.session_id:
        pass
    else:
        os.chdir(os.path.join(qsiprep_dir, temp_participant))
        sessions = glob.glob("ses-*")
        if len(sessions) == 0:
            sessions = None

    for temp_session in sessions:

        if temp_session is not None:
            temp_session = temp_session.split("-")[1]

        create_gifs(
            qsiprep_dir,
            temp_participant.split("-")[-1],
            output_dir,
            session=temp_session,
        )

# CreateTiledMosaic -i dwi_res_anat_file.nii -r sub-164209/ses-V02/dwi/sub-164209_ses-V02_run-1_space-T1w_desc-preproc_dwi_L1_DT_DEC.nii -a 0.45 -x dwi_res_brainmask.nii -t 1x5 -o mosiac1.png -f 0x1 -s 20x24x28x30x32
