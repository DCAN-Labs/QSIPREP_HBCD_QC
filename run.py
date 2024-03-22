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
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.special import expit

warnings.simplefilter(action="ignore", category=FutureWarning)

# Settings for the gifs
# Specify the slices that we would like to save as a fraction
# of the masked extent.
slice_ratios = [
    np.array([0.2, 0.48, 0.8]),  # LR Slices
    np.array([0.4, 0.6]),  # AP Slices
    np.array([0.2, 0.7]),  # IS Slices
]
slice_names = [
    [
        "LeftTemporalSagittal",
        "LeftBrainStemSagittal",
        "RightTemporalSagittal",
    ],
    [
        "AnteriorCoronal",
        "PosteriorCoronal",
    ],
    [
        "CerebellumAxial",
        "SemiovaleAxial",
    ],
]
FA_THRESHOLD = 0.02
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
        check=True,
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
        check=True,
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
            pngres_mask,
            "-o",
            pngres_anat,
            "-v",
            "1",
            "--interpolation",
            "NearestNeighbor",
        ],
        check=True,
    )

    # Scale it nicely so the DEC pops
    subprocess.run(
        [
            "ImageMath",
            "3",
            pngres_anat,
            "TruncateImageIntensity",
            pngres_anat,
            "0.02",
            "0.98",
            "256",
        ],
        check=True,
    )

    # Clean up
    os.remove("_mask_box.nii")

    return pngres_mask, pngres_anat


def mask_fa(fa_image):
    """Turn an FA image into a mask by overwriting it."""
    fa_img = nb.load(fa_image)
    nb.Nifti1Image(
        (fa_img.get_fdata() > FA_THRESHOLD).astype(np.float32),
        fa_img.affine
    ).to_filename(fa_image)


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
    dwires_anat_mask = op.abspath("dwires_anat_mask.nii")
    pngres_fa_mask = op.abspath("pngres_fa_mask.nii")

    # Unzip the dwi file
    subprocess.run(["gunzip", zipped_preproc_dwi_file])
    preproc_dwi_file = op.abspath(zipped_preproc_dwi_file.replace(".gz", ""))
    fstem = preproc_dwi_file.replace(".nii", "")

    # Resample the mask into the dwi space
    subprocess.run(
        [
            "antsApplyTransforms",
            "-d",
            "3",
            "--interpolation",
            "NearestNeighbor",
            "-o",
            dwires_anat_mask,
            "-i",
            pngres_mask,
            "-r",
            preproc_dwi_file,
        ],
        check=True,
    )

    # Convert bval, bvec to bmat
    subprocess.run(["FSLBVecsToTORTOISEBmatrix", f"{fstem}.bval", f"{fstem}.bvec"])

    # Estimate the tensor (the bmtxt is automatically found)
    subprocess.run(
        [
            "EstimateTensor",
            "--input",
            preproc_dwi_file,
            "--mask",
            dwires_anat_mask,
        ],
        check=True,
    )
    # Calculate FA
    subprocess.run(["ComputeFAMap", f"{fstem}_L1_DT.nii", "1"], check=True)

    # Make the DEC Map
    subprocess.run(["ComputeDECMap", "--input_tensor", f"{fstem}_L1_DT.nii", "--useFA"], check=True)

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
            f"{fstem}_L1_DT_DEC.nii",
            "-r",
            pngres_mask,
        ],
        check=True,
    )

    # Resample the FA Map into pngres space
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
            pngres_fa_mask,
            "-i",
            f"{fstem}_L1_DT_FA.nii",
            "-r",
            pngres_mask,
        ],
        check=True,
    )

    # Turn the FA Map into a mask
    mask_fa(pngres_fa_mask)

    # Clean up
    os.remove(f"{fstem}_L1_DT_DEC.nii")
    os.remove(f"{fstem}_L1_DT.nii")
    os.remove(f"{fstem}_L1_DT_FA.nii")
    os.remove(f"{fstem}_L1_AM.nii")
    os.remove(dwires_anat_mask)
    # os.remove(preproc_dwi_file)  # This is the unzipped nii, the original stays

    return pngres_dec, pngres_fa_mask


def get_anchor_slices_from_mask(mask_file, axis=2):
    """Find the slice numbers for ``slice_ratios`` inside a mask."""
    mask_arr = nb.load(mask_file).get_fdata()
    mask_coords = np.argwhere(mask_arr > 0)[:, axis]
    min_slice = mask_coords.min()
    max_slice = mask_coords.max()
    covered_slices = max_slice - min_slice
    return np.floor(
        (covered_slices * slice_ratios[axis]) + min_slice).astype(np.int64)


def gifs_from_dec(dec_file, mask_file, anat_file, prefix):
    """Create a series of animated gifs from DEC+anatomical images."""

    # Loop is for individual slices in the gif image
    for axis in [0, 1, 2]:
        # Compute the indices of the slices
        slice_indices = get_anchor_slices_from_mask(mask_file, axis)
        axis_slice_names = slice_names[axis]
        named_slices = zip(slice_indices, axis_slice_names)
        for base_slice_idx, slice_name in named_slices:
            output_gif_path = f"{prefix}{slice_name}.gif"
            images = []
            image_files = []
            for offset_idx, slice_offset in enumerate(slice_gif_offsets):
                slice_idx = base_slice_idx + slice_offset
                slice_png_path = f"{prefix}{slice_name}_part-{offset_idx}_dec.png"
                cmd = [
                    "CreateTiledMosaic",
                    "-i",
                    anat_file,
                    "-r",
                    dec_file,
                    "-a",
                    "0.65",
                    "-x",
                    mask_file,
                    "-t",
                    "1x1",
                    "-o",
                    slice_png_path,
                    "-s",
                    f"{slice_idx}x{slice_idx}",
                    "-d",
                    f"{axis}"
                ]

                if axis in (0, 1):
                    cmd += ["-f", "0x1"]

                # Run CreateTiledMosaic to get a png
                subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                )

                # Upsample the image ONCE
                ants_png = Image.open(slice_png_path)
                resized = ants_png.resize((512, 512), Image.NEAREST)
                images.append(resized)
                image_files.append(slice_png_path)

            # Create a back and forth animation by appending the images to
            # themselves, but reversed
            images = images + images[-2:0:-1]

            # Save the gif
            imageio.mimsave(
                output_gif_path, images, loop=0, duration=(1 / fps) * 1000, subrectangles=True
            )

            # Clean up the temp pngs
            for imagef in image_files:
                os.remove(imagef)


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
        **initial_bids_filters,
    )
    if not anat_hires_file:
        raise Exception(f"No high-res T2w image found for f{subject}")
    anat_hires_file = anat_hires_file[0]
    pngres_brain_mask, pngres_anat = create_png_space(anat_mask_file, anat_hires_file)

    # Create the local output dir
    if session is not None:
        session_name = f"ses-{session}"
    else:
        session_name = ""
    png_dir = op.join(output_dir, f"sub-{subject}/{session_name}/dwi").replace(
        "//", "/"
    )
    os.makedirs(png_dir, exist_ok=True)

    # The anatomical outputs from QSIPrep v<1.0
    # cannot be in session directories
    # Add this filter after we've found the anatomical data
    if session is not None:
        initial_bids_filters["session"] = session

    # Find all the dwi files, and their corresponding dwi files
    dwi_files = layout.get(suffix="dwi", extension="nii.gz", **initial_bids_filters)
    for dwi_file in dwi_files:
        pngres_dwi, pngres_fa_mask = dec_in_pngspace(pngres_brain_mask, dwi_file)
        gif_prefix = op.join(
            png_dir, f"sub-{subject}_{session_name}_qcgif-".replace("__", "_")
        )
        print(f"Creating GIFs for {dwi_file}")
        gifs_from_dec(pngres_dwi, pngres_fa_mask, pngres_anat, prefix=gif_prefix)
        print("Done")


def ritchie_fa_plot(dec_file, fa_file, anat_file, slice_indices):

    # Loop is for individual slices in the gif image
    for gif_idx, base_slice_idx in enumerate(slice_indices):
        images = []
        for offset_idx, slice_offset in enumerate(slice_gif_offsets):
            slice_idx = base_slice_idx + slice_offset

            fig, ax = plt.subplots(1, 1, figsize=my_figsize)

            slice_anat = ndimage.rotate(b0_data[:, :, slice_idx], -90)
            slice_rgb = ndimage.rotate(RGB[:, :, slice_idx], -90)

            fa_slice = FA_masked[:, :, slice_idx]
            xmax = 5
            trans_x = -xmax + 2 * xmax * (fa_slice + 0.1)
            fa_slice = expit(trans_x)

            alpha = ndimage.rotate(np.array(255 * fa_slice, "uint8"), -90)[:, :, np.newaxis]
            slice_rgba = np.concatenate([slice_rgb, alpha], axis=-1)

            _ = ax.imshow(slice_anat, cmap=plt.cm.Greys_r)
            _ = ax.imshow(slice_rgba)
            _ = ax.axis("off")

            file_path = op.join(
                png_dir,
                fname_gif + str(gif_idx) + "_" + str(offset_idx) + ".png"
            )

            fig.savefig(file_path, bbox_inches="tight")
            plt.close(fig)


if __name__ == "__main__":
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
        if (
            len(glob.glob(dwi_path_sessions)) + len(glob.glob(dwi_path_no_sessions))
            == 0
        ):
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
