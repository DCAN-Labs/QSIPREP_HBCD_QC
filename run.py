#!/usr/local/miniconda/bin/python
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
from scipy.stats import scoreatpercentile
from dipy.io.image import load_nifti, save_nifti
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
import dipy.reconst.dti as dti


warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)
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

# Specify the image dimensions, note that SwipesForScience images should be square
# img_height = 4.8
# aspect_ratio = 1.0
# img_width = aspect_ratio * img_height
# figsize = (img_width, img_height)
# figsize_multiplier = 1.5
my_figsize = (7.2, 7.2)

FA_ALPHA_RANGE = 5  # Plus/minus range for epit
FA_ALPHA_OFFSET = 0.17  # Added to FA
FA_ALPHA_MULT = 4  # Steepness of expit
BRIGHTNESS_UPSCALE = 2.0

PNGRES_SIZE = 90


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


def create_dwires_png_space(dwi_file, hires_maskfile, hires_anatfile, res="dwi"):
    """Create an output space with dwi voxel size that is ideal for
    creating square png images.

    Parameters
    ----------

    dwi_file: str
        Path to qsiprep-preprocessed dwi file

    hires_maskfile: str
        Path to the brain mask from qsiprep. If res is "anat" it should be a
        cube already

    hires_anatfile: str
        Path to the brain mask from qsiprep. If res is "anat" it should be a
        cube already

    res : str
        "dwi" or "anat" - which resolution to do calculations in

    Returns
    -------

    pngres_maskfile: str
        Path to the brain mask in png space

    pngres_anatfile: str
        Path to the anatomical image in png space

    """

    # Create a cube bounding box that we will use to take pics
    pngres_dwi = op.abspath(f"{res}pngres_dwi.nii")
    pngres_anat = op.abspath(f"{res}pngres_anat.nii")
    pngres_dec = op.abspath(f"{res}pngres_dec.nii")
    pngres_fa = op.abspath(f"{res}pngres_fa.nii")
    pngres_mask = op.abspath(f"{res}pngres_mask.nii")
    fstem = dwi_file.replace(".nii.gz", "")

    if res == "dwi":
        # Zeropad the DWI so it's a cube
        subprocess.run(
            [
                "3dZeropad",
                "-RL",
                f"{PNGRES_SIZE}",
                "-AP",
                f"{PNGRES_SIZE}",
                "-IS",
                f"{PNGRES_SIZE}",
                "-prefix",
                pngres_dwi,
                dwi_file,
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
                hires_maskfile,
                "-r",
                pngres_dwi,
                "-o",
                pngres_mask,
                "-v",
                "1",
                "--interpolation",
                "NearestNeighbor",
            ],
            check=True,
        )

        # Resample the anat file into pngres
        subprocess.run(
            [
                "antsApplyTransforms",
                "-d",
                "3",
                "-i",
                hires_anatfile,
                "-r",
                pngres_dwi,
                "-o",
                pngres_anat,
                "-v",
                "1",
                "--interpolation",
                "NearestNeighbor",
            ],
            check=True,
        )
    elif res == "anat":
        subprocess.run(
            [
                "antsApplyTransforms",
                "-d",
                "3",
                "-e",
                "3",
                "-i",
                dwi_file,
                "-r",
                hires_maskfile,
                "-o",
                pngres_dwi,
                "-v",
                "1",
                "--interpolation",
                "BSpline",
            ],
            check=True,
        )
        pngres_mask = hires_maskfile
        pngres_anat = hires_anatfile

    # Use DIPY to fit a tensor
    data, affine = load_nifti(pngres_dwi)
    mask_data, _ = load_nifti(pngres_mask)
    bvals, bvecs = read_bvals_bvecs(f"{fstem}.bval", f"{fstem}.bvec")
    gtab = gradient_table(bvals, bvecs)
    tenmodel = dti.TensorModel(gtab)
    print(f"Fitting Tensor to {pngres_dwi}")
    tenfit = tenmodel.fit(data, mask=mask_data > 0)

    # Get FA and DEC from the tensor fit
    FA = dti.fractional_anisotropy(tenfit.evals)
    FA = np.clip(FA, 0, 1)

    # Convert to colorFA image as in DIPY documentation
    FA_masked = FA * mask_data
    RGB = dti.color_fa(FA_masked, tenfit.evecs)
    RGB = np.array(255 * RGB, 'uint8')
    save_nifti(pngres_fa, FA_masked.astype(np.float32), affine)
    save_nifti(pngres_dec, RGB, affine)

    return pngres_dec, pngres_fa, pngres_anat, pngres_mask


def create_hires_png_space(hires_maskfile, hires_anatfile):
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

    # Make the DEC Map. Not using --useFA because FA will appear in the alpha channel
    subprocess.run(
        [
            "ComputeDECMap",
            "--input_tensor",
            f"{fstem}_L1_DT.nii",
            "--useFA",
            "--color_scalexp",
            "0.3",
        ],
        check=True,
    )

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

    # Clean up
    os.remove(f"{fstem}_L1_DT_DEC.nii")
    os.remove(f"{fstem}_L1_DT.nii")
    os.remove(f"{fstem}_L1_DT_FA.nii")
    os.remove(f"{fstem}_L1_AM.nii")
    os.remove(dwires_anat_mask)
    # os.remove(preproc_dwi_file)  # This is the unzipped nii, the original stays

    return pngres_dec, pngres_fa_mask


def get_anchor_slices_from_mask(mask_file, axis):
    """Find the slice numbers for ``slice_ratios`` inside a mask."""
    mask_arr = nb.load(mask_file).get_fdata()
    mask_coords = np.argwhere(mask_arr > 0)[:, axis]
    min_slice = mask_coords.min()
    max_slice = mask_coords.max()
    covered_slices = max_slice - min_slice
    return np.floor((covered_slices * slice_ratios[axis]) + min_slice).astype(np.int64)


def load_and_rotate_png(pngfile, axis):
    img = Image.open(pngfile)
    if axis == 0:
        return img.transpose(Image.ROTATE_90)
    if axis == 1:
        return img.transpose(Image.ROTATE_90)
    if axis == 2:
        return img.transpose(Image.ROTATE_270)


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
                    f"{axis}",
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
                output_gif_path,
                images,
                loop=0,
                duration=(1 / fps) * 1000,
                subrectangles=True,
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
    if session:
        initial_bids_filters["session"] = session.replace("ses-", "")

    # Prepare the PNG space and get the anatomical data resampled into it
    anat_mask_file = layout.get(
        suffix="mask",
        datatype="anat",
        extension="nii.gz",
        space="ACPC",
        **initial_bids_filters,
    )
    if not anat_mask_file:
        raise Exception(f"No anatomical brain mask produced for {subject}")
    anat_mask_file = anat_mask_file[0]
    # We get a T2w for HBCD
    anat_hires_file = layout.get(
        suffix="T2w",
        datatype="anat",
        extension=".nii.gz",
        **initial_bids_filters,
    )
    if not anat_hires_file:
        raise Exception(f"No high-res T2w image found for f{subject}")
    anat_hires_file = anat_hires_file[0]

    # Create the local output dir
    if session is not None:
        session_name = f"ses-{session}"
    else:
        session_name = ""
    png_dir = op.join(output_dir, f"sub-{subject}/{session_name}/dwi").replace(
        "//", "/"
    )
    os.makedirs(png_dir, exist_ok=True)

    # get the hires anatomical data into a nice cube
    hirescube_mask_file, hirescube_anat_file = create_hires_png_space(anat_mask_file, anat_hires_file)

    # Find all the dwi files, and their corresponding dwi files
    dwi_files = layout.get(suffix="dwi", extension="nii.gz", **initial_bids_filters)
    for dwi_file in dwi_files:
        print(f"Creating GIFs for {dwi_file}")
        pngres_dec, pngres_fa, pngres_anat, pngres_brain_mask = create_dwires_png_space(
            dwi_file,
            anat_mask_file,
            anat_hires_file,
            res="dwi",
        )

        # Do the low-res version
        gif_prefix = op.join(
            png_dir, f"sub-{subject}_{session_name}_res-dwi_qcgif-".replace("__", "_")
        )
        richie_fa_gifs(pngres_dec, pngres_fa, pngres_anat, pngres_brain_mask, gif_prefix)

        # Do the hi-res version
        print(f"Creating Hi-Res GIFs for {dwi_file}")
        pnghires_dec, pnghires_fa, _, _ = create_dwires_png_space(
            dwi_file,
            hirescube_mask_file,
            hirescube_anat_file,
            res="anat"
        )
        hires_gif_prefix = op.join(
            png_dir, f"sub-{subject}_{session_name}_res-anat_qcgif-".replace("__", "_")
        )
        richie_fa_gifs(pnghires_dec, pnghires_fa, hirescube_anat_file, hirescube_mask_file, hires_gif_prefix)

        print("Done")


def fa_to_alpha(
    fa_data,
):
    return expit(
        -FA_ALPHA_RANGE + FA_ALPHA_MULT * FA_ALPHA_RANGE * (fa_data + FA_ALPHA_OFFSET)
    )


def richie_fa_gifs(dec_file, fa_file, anat_file, mask_file, prefix):
    """Create GIFs for Swipes using ARH's method from HBN-POD2.

    Parameters
    ----------

    dec_file: str
        Path to an RGB DEC NIFTI file resampled into pngres space

    fa_file: str
        Path to a NIFTI file of FA values in pngres space

    anat_file: str
        Path to anatomical file to display in grayscale behind the RBGA data.
        Also must be in pngres space

    mask_file: str
        Path to the brainmask NIFTI file in pngres space

    prefix: str
        Stem of the gifs that will be written

    Returns: None

    """

    anat_img = nb.load(anat_file)
    anat_data = anat_img.get_fdata()
    anat_vmin, anat_vmax = scoreatpercentile(anat_data.flatten(), [1, 98])

    # Load the RGB data. It was created by tortoise, but resampled into
    # pngres space. This also converts it to a 3-vector data type.
    rgb_img = nb.load(dec_file)
    rgb_data = rgb_img.get_fdata().squeeze()
    rgb_data = np.clip(0, 255, rgb_data * BRIGHTNESS_UPSCALE)

    # Open FA image and turn it into alpha values
    fa_img = nb.load(fa_file)
    fa_data = fa_to_alpha(np.clip(0, 1, fa_img.get_fdata())) * 255

    print(f"Setting grayscale vmax to {anat_vmax}")

    def get_anat_rgba_slices(idx, axis):
        # Select slice from axis and handle rotation
        if axis == 0:
            anat = anat_data[idx, :, :]
            rgb = rgb_data[idx, :, :]
            fa = fa_data[idx, :, :]
        elif axis == 1:
            anat = anat_data[:, idx, :]
            rgb = rgb_data[:, idx, :]
            fa = fa_data[:, idx, :]
        else:
            anat = anat_data[:, :, idx]
            rgb = rgb_data[:, :, idx]
            fa = fa_data[:, :, idx]
        rgba = np.concatenate([rgb, fa[:, :, np.newaxis]], axis=-1)
        return anat, rgba

    # Make the gifs!
    temp_image_files = []
    for axis in [0, 1, 2]:
        # Compute the indices of the slices
        slice_indices = get_anchor_slices_from_mask(mask_file, axis)
        axis_slice_names = slice_names[axis]
        named_slices = zip(slice_indices, axis_slice_names)

        for base_slice_idx, slice_name in named_slices:
            output_gif_path = f"{prefix}{slice_name}.gif"
            images = []

            for offset_idx, slice_offset in enumerate(slice_gif_offsets):
                slice_idx = base_slice_idx + slice_offset
                slice_png_path = f"{prefix}{slice_name}_part-{offset_idx}_dec.png"
                fig, ax = plt.subplots(1, 1, figsize=my_figsize)
                slice_anat, slice_rgba = get_anat_rgba_slices(slice_idx, axis)

                _ = ax.imshow(
                    slice_anat,
                    vmin=anat_vmin,
                    vmax=anat_vmax,
                    cmap=plt.cm.Greys_r,
                )
                _ = ax.imshow(slice_rgba.astype(np.uint8))
                _ = ax.axis("off")

                fig.savefig(slice_png_path, bbox_inches="tight")
                plt.close(fig)
                images.append(load_and_rotate_png(slice_png_path, axis))
                temp_image_files.append(slice_png_path)

            # Create a back and forth animation by appending the images to
            # themselves, but reversed
            images = images + images[-2:0:-1]

            # Save the gif
            imageio.mimsave(
                output_gif_path,
                images,
                loop=0,
                duration=(1 / fps) * 1000,
                subrectangles=True,
            )

    # Clean up the
    for temp_image in temp_image_files:
        os.remove(temp_image)


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
