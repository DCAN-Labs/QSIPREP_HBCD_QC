#!/usr/local/bin/python3

## https://github.com/richford/fibr/blob/master/notebooks/2020-11-16-cloudknot-create-color-fa-gifs.ipynb
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import bids
import dipy.reconst.dti as dti
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os, glob
import os.path as op
import argparse

from dipy.io.image import load_nifti, save_nifti
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from scipy import ndimage
from scipy.special import expit


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("qsiprep_dir", help="The path to the BIDS directory for your study (this is the same for all subjects)", type=str)
    parser.add_argument("output_dir", help="The path to the folder where outputs will be stored (this is the same for all subjects)", type=str)
    parser.add_argument("analysis_level", help="Should always be participant", type=str)
    
    parser.add_argument('--participant_label', '--participant-label', help="The name/label of the subject to be processed (i.e. sub-01 or 01)", type=str)
    parser.add_argument('--session_id', '--session-id', help="OPTIONAL: the name of a specific session to be processed (i.e. ses-01)", type=str)    
    parser.add_argument('--bids_directory', '--bids_directory', help="OPTIONAL: This is not actually used for processing.", type=str)  

    return parser.parse_args()

def create_gifs(bids_dir, subject, output_dir, session = None):
    
    # Download that particular subject to a local folder
    layout = bids.BIDSLayout(bids_dir, validate=False)

    # Specify the slices that we would like to save as a fraction of total number of slices
    scale_fa = True
    slice_ratios = np.linspace(0.4, 0.6, 4, endpoint=True)
    slice_ratios[0] = 0.42
    slice_ratios[-1] = 0.58
    
    # Specify that slice range for the animated gifs
    slice_gif_offsets = np.arange(-5, 6)
    
    # Calculate the frames-per-second for the animated gifs
    fps = len(slice_gif_offsets) / 2.0
    
    # Specify the image dimensions, note that SwipesForScience images should be square
    img_height = 4.8
    aspect_ratio = 1.0
    img_width = aspect_ratio * img_height
    figsize = (img_width, img_height)

    # Specify local filenames
    subject = subject.replace("sub-", "")
    fname_gif = f"sub-{subject}_desc-b0colorfa_slice-"
    fname_fa = f"sub-{subject}_tensor_fa.nii.gz"
    fname_rgb = f"sub-{subject}_tensor_rgb.nii.gz"
    
    # Use pybids to grab the necessary image files
    initial_bids_filters = {"subject": subject, "return_type": "filename"}
    if type(session) != type(None):
        initial_bids_filters["session"] = session
    fb0s = layout.get(suffix="dwiref", extension="nii.gz", **initial_bids_filters)
    for i, temp_fb0 in enumerate(fb0s):

        temp_fb0_split = temp_fb0.split('/')[-1].split('_')[:-1]
        temp_entities = {}
        for temp_split in temp_fb0_split:
            temp_split_split = temp_split.split('-')
            temp_entities[temp_split_split[0]] = temp_split_split[1]
        if 'run' in temp_entities.keys():
            run_specific_bids_filters = initial_bids_filters.copy()
            run_specific_bids_filters['run'] = temp_entities['run']
            fname_gif = f"sub-{subject}_run-{temp_entities['run']}_desc-b0colorfa_slice-"
            fname_fa = f"sub-{subject}_run-{temp_entities['run']}_tensor_fa.nii.gz"
            fname_rgb = f"sub-{subject}_run-{temp_entities['run']}_tensor_rgb.nii.gz"
        else:
            fname_gif = f"sub-{subject}_desc-b0colorfa_slice-"
            fname_fa = f"sub-{subject}_tensor_fa.nii.gz"
            fname_rgb = f"sub-{subject}_tensor_rgb.nii.gz"
    
        fb0 = layout.get(suffix="dwiref", extension="nii.gz", **run_specific_bids_filters)[0]
        fdwi= layout.get(suffix="dwi", extension="nii.gz", **run_specific_bids_filters)[0]
        fmask = layout.get(suffix="mask", datatype="dwi", extension="nii.gz", **run_specific_bids_filters)[0]
    
        # Load the niftis
        b0_data, b0_affine = load_nifti(fb0)
        t1w_data, t1w_affine = load_nifti(fb0)
        mask_data, mask_affine = load_nifti(fmask)
        data, affine = load_nifti(fdwi)
        
        # Load the gradient table
        fbval = layout.get_bval(path=fdwi, subject=subject)
        fbvec = layout.get_bvec(path=fdwi, subject=subject)
        bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
        gtab = gradient_table(bvals, bvecs)

        # Fit a tensor model and compute FA
        tenmodel = dti.TensorModel(gtab)
        tenfit = tenmodel.fit(data)
        FA = dti.fractional_anisotropy(tenfit.evals)
        FA = np.clip(FA, 0, 1)

        # Convert to colorFA image as in DIPY documentation
        FA_masked = FA * mask_data
        RGB = dti.color_fa(FA_masked, tenfit.evecs)

        RGB = np.array(255 * RGB, 'uint8')
        save_nifti(fname_fa, FA_masked.astype(np.float32), affine)
        save_nifti(fname_rgb, RGB, affine) 
    
    def trim_zeros(arr, margin=0, trim_dims=None):
        '''
        Trim the leading and trailing zeros from a N-D array.

        :param arr: numpy array
        :param margin: how many zeros to leave as a margin
        :returns: trimmed array
        :returns: slice object
        '''
        s = []
        if trim_dims is None:
            trim_dims = list(range(arr.ndim))

        for dim in range(arr.ndim):
            start = 0
            end = -1

            if dim in trim_dims:
                slice_ = [slice(None)]*arr.ndim

                go = True
                while go:
                    slice_[dim] = start
                    go = not np.any(arr[tuple(slice_)])
                    start += 1
                start = max(start-1-margin, 0)

                go = True
                while go:
                    slice_[dim] = end
                    go = not np.any(arr[tuple(slice_)])
                    end -= 1
                end = arr.shape[dim] + min(-1, end+1+margin) + 1

                s.append(slice(start,end))
            else:
                s.append(slice(None, None, None))
        return arr[tuple(s)], tuple(s)
        
    def pad_square_2d(arr):
        """Pad a slice so that it is square"""
        dim_x, dim_y = arr.shape[0], arr.shape[1]
        dim_max = max(dim_x, dim_y)
        pad_xr = (dim_max - dim_x) // 2
        pad_xl = dim_max - dim_x - pad_xr
        pad_yr = (dim_max - dim_y) // 2
        pad_yl = dim_max - dim_y - pad_yr

        pad_width = [(pad_xl, pad_xr), (pad_yl, pad_yr)]
        for i in range(arr.ndim - 2):
            pad_width.append((0, 0))
        return np.pad(arr, pad_width=pad_width)

    # Compute the indices of the slices
    slice_indices = np.array(slice_ratios * b0_data.shape[-1], dtype="uint8")

    # Trim zeros off of everything
    mask_trim, trim_slices = trim_zeros(mask_data, margin=5, trim_dims=(0, 1))
    t1w_data = t1w_data[trim_slices]
    b0_data = b0_data[trim_slices]
    RGB = RGB[trim_slices + (slice(None, None, None),)]
    FA_masked = FA_masked[trim_slices]
    
    # Square everything
    t1w_data = pad_square_2d(t1w_data)
    RGB = pad_square_2d(RGB)
    FA_masked = pad_square_2d(FA_masked)
    b0_data = pad_square_2d(b0_data)

    # Create the local output dir
    if type(session) != type(None):
        session_name = session
    else:
        session_name = ''
    png_dir = os.path.join(output_dir, 'sub-' + subject, 'ses-' + session_name, 'dwi')
    os.makedirs(png_dir, exist_ok=True)
    
    # Enlarge images
    figsize_multiplier = 1.5
    my_figsize = tuple(x * figsize_multiplier for x in figsize)
    
    # Loop is for individual slices in the gif image
    for gif_idx, base_slice_idx in enumerate(slice_indices):
        images = []
        for offset_idx, slice_offset in enumerate(slice_gif_offsets):
            slice_idx = base_slice_idx + slice_offset

            fig, ax = plt.subplots(1, 1, figsize=my_figsize)

            slice_anat = ndimage.rotate(b0_data[:, :, slice_idx], -90)
            slice_rgb = ndimage.rotate(RGB[:, :, slice_idx], -90)

            fa_slice = FA_masked[:, :, slice_idx]
            if scale_fa:
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

            images.append(imageio.imread(file_path))

        images = images + images[-2:0:-1]

        file_path = op.join(
            png_dir,
            fname_gif + str(gif_idx) + ".gif"
        )
        
        # Save the gif
        imageio.mimsave(file_path, images, loop=0, duration=(1/fps)*1000, subrectangles=True)
            

#Grab the arg parse inputs 
args = parse_args()
cwd = os.getcwd()

#reassign variables to command line input
qsiprep_dir = args.qsiprep_dir
if os.path.isabs(qsiprep_dir) == False:
    qsiprep_dir = os.path.join(cwd, qsiprep_dir)
output_dir = args.output_dir
if os.path.isabs(output_dir) == False:
    output_dir = os.path.join(cwd, output_dir)
analysis_level = args.analysis_level
if analysis_level != 'participant':
    raise ValueError('Error: analysis level must be participant, but program received: ' + analysis_level)

if args.participant_label:
    pass
else:
    os.chdir(qsiprep_dir)
    participants = glob.glob('sub-*')

for temp_participant in participants:

    #Find sessions... if session was provided then
    #process that specific session. Otherwise iterate
    #through all sessions or continue without assumption
    #of sessions if no sessions are found in the BIDS
    #structure

    #This is mainly to weed out html reports...
    dwi_path_sessions = os.path.join(qsiprep_dir, temp_participant, 'ses*', 'dwi')
    dwi_path_no_sessions = os.path.join(qsiprep_dir, temp_participant, 'ses*')
    if len(glob.glob(dwi_path_sessions)) + len(glob.glob(dwi_path_no_sessions)) == 0:
        continue


    if args.session_id:
        pass
    else:
        os.chdir(os.path.join(qsiprep_dir, temp_participant))
        sessions = glob.glob('ses-*')
        if len(sessions) == 0:
            sessions = None

    for temp_session in sessions:

        if type(temp_session) != type(None):
            temp_session = temp_session.split('-')[1]

        create_gifs(qsiprep_dir, temp_participant.split('-')[-1], output_dir, session = temp_session)
