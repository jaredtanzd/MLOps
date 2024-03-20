# Generate mask (brain, ventricles, wm/gm/csf)

import os
import numpy as np
import nibabel as nib

from PIL import Image

import imageio

def scale_to_255(mask):
    mask_nd = np.array(mask).astype(np.uint8)
    mask_255 = np.where(mask_nd == 1, 255, 0)
    mask_new = Image.fromarray(mask_255.astype(np.uint8))
    return mask_new

def normalize(volume, hu):
    """Normalize the volume"""
    hu_min, hu_max = hu
    print(hu_min, hu_max, hu_max-hu_min)
    
    # window
    volume[volume < hu_min] = hu_min
    volume[volume > hu_max] = hu_max
    
    # normalise (min-max to 0-1)
    # volume = (volume - hu_min) / (hu_max - hu_min)
    return volume

src = '/data/data_repo/neuro_img/anat_brain_img/datasets/CERMEP-IDB-MRXFDG/processed/yihao/freesurfer/' # freesurfer
tgt = '/data/data_repo/neuro_img/anat_brain_img/datasets/CERMEP-IDB-MRXFDG/processed/yihao/labels_new_nonorm/' # labels_all labels_norm

os.makedirs(tgt + 'ct/', exist_ok=True)
os.makedirs(tgt + 'ct_norm/', exist_ok=True)
os.makedirs(tgt + 'mri/', exist_ok=True)
os.makedirs(tgt + 'mri_norm/', exist_ok=True)

os.makedirs(tgt + 'vent/', exist_ok=True)
os.makedirs(tgt + 'bet/', exist_ok=True)
os.makedirs(tgt + 'tissue/', exist_ok=True)
os.makedirs(tgt + 'csf/', exist_ok=True)
os.makedirs(tgt + 'gm/', exist_ok=True)
os.makedirs(tgt + 'wm/', exist_ok=True)

os.makedirs(tgt + 'vent_alt/', exist_ok=True)
os.makedirs(tgt + 'bet_alt/', exist_ok=True)
os.makedirs(tgt + 'tissue_alt/', exist_ok=True)
os.makedirs(tgt + 'csf_alt/', exist_ok=True)
os.makedirs(tgt + 'gm_alt/', exist_ok=True)
os.makedirs(tgt + 'wm_alt/', exist_ok=True)

for sub_full in os.listdir(src):
    
    sub = sub_full.split('_')[0]
    
    print()
    print()
    print(sub, sub_full)    
    
    # 0. Data
    
    ct_raw = nib.load(src + '../MNI/' + sub + '/' + sub + '_space-MNI_ct.nii.gz') # this is OK
    ct = ct_raw.get_fdata()
    print('ct', ct.shape)
    
    ## window CT and then normalize to 0-1
    ct_norm = np.nan_to_num(ct.copy())
    ct_norm = normalize(ct_norm, hu=[0,80]) # brain window
    
    
    # mri_raw = nib.load(src + '../MNI/' + sub + '/' + sub + '_space-MNI_T1w.nii.gz') # currently it reads in the raw T1. Should change this to the processed T1. read mgz directly.
    mri_raw = nib.load(src + sub_full + '/mri/T1-in-rawavg.mgz') # currently it reads in the raw T1. Should change this to the processed T1. read mgz directly.
    mri = mri_raw.get_fdata()
    print('mri', mri.shape)
    
    ## normalise to 0-1
    mri_min = mri.min()
    mri_max = mri.max()
    mri_norm = (mri - mri_min)/(mri_max - mri_min)
    
    x,y,z = mri.shape
    voxel_total = x * y * z
    
    # 1. Ventricle Segmentation mask
    
    ventri_raw = nib.load(src + sub_full + '/mri/aparc.a2009s+aseg-in-rawavg.mgz')
    ventri = ventri_raw.get_fdata()
    print('ventri', ventri.shape)
    
    num_brain_voxels = np.where(ventri!=0)[0].shape[0]
    print(100*num_brain_voxels / voxel_total, '% of image volume has brains.', num_brain_voxels, 'voxels out of', voxel_total)

    total = 0
    ventri_idx = [4, 5, 14, 15, 43, 44]
    for i in ventri_idx:
        idx = np.where(ventri==i)
        total += 100 * idx[0].shape[0] / num_brain_voxels
        
    print(total, '% of brain are ventricles')
    
    # ventri_all is the brain volume with ventricles labelled as 1
    ventri_all = np.where(
        (ventri == 4) | (ventri == 5) | (ventri == 14) | (ventri == 15) | (ventri == 43) | (ventri == 44),
        1,0
    )

    print(ventri_all.sum(), 'voxels,', ventri_all.sum() / voxel_total, '% of total') 
    
    
    # 2. Brain mask
    
    brain_raw = nib.load(src + sub_full + '/mri/brainmask-in-rawavg.mgz')
    brain_data = brain_raw.get_fdata()
    print('brain_data', brain_data.shape)
    
    brain_data_bin = np.where(brain_data!=0, 1, 0)
    
    # brain_data_bin is the brain volume with brain matter labelled as 1
    print('brain mask', brain_data_bin.sum(), 100*brain_data_bin.sum()/voxel_total, '% of total')
    
    
    # 3. Tissue Segmentation mask
    
    raw_tissue = nib.load(
        src + '../MRI/' + sub + '_space-MNI_T1w_brain_seg.nii.gz' # from FSL FAST output, based on brain.mgz from freesurfer
    )
    data_tissue = raw_tissue.get_fdata()
    print('data_tissue', data_tissue.shape)
    
    tissue = np.where(data_tissue!=0, 1, 0)
    csf = np.where(data_tissue==1, 1, 0)
    gm = np.where(data_tissue==2, 1, 0)
    wm = np.where(data_tissue==3, 1, 0)
    
    print('csf mask', csf.sum(), 100*csf.sum()/voxel_total)
    print('gm mask', gm.sum(), 100*gm.sum()/voxel_total)
    print('wm mask', wm.sum(), 100*wm.sum()/voxel_total)
    
    print('tissue mask', tissue.sum(), 100*tissue.sum()/voxel_total)

    # Generate (axial) slices, skip slices without brain 
    
    for i in range(ct_norm.shape[-1]): # axial
        
        brain_slice = ct_norm[:,:,i]
        num_unique_vals = np.unique(brain_slice).shape[0]
        
        if num_unique_vals != 1: # if corresponding ct slice no brain, skip
            
            fname = tgt + 'vent_alt/' + sub + '_' + str(i) + '.png'
            imageio.imwrite(fname, ventri_all[::-1,::-1,i].T.astype(np.uint8))
            
            fname = tgt + 'bet_alt/' + sub + '_' + str(i) + '.png'
            imageio.imwrite(fname, brain_data_bin[::-1,::-1,i].T.astype(np.uint8))
            
            fname = tgt + 'tissue_alt/' + sub + '_' + str(i) + '.png'
            imageio.imwrite(fname, tissue[::-1,::-1,i].T.astype(np.uint8))
            
            fname = tgt + 'csf_alt/' + sub + '_' + str(i) + '.png'
            imageio.imwrite(fname, csf[::-1,::-1,i].T.astype(np.uint8))
            
            fname = tgt + 'gm_alt/' + sub + '_' + str(i) + '.png'
            imageio.imwrite(fname, gm[::-1,::-1,i].T.astype(np.uint8))
            
            fname = tgt + 'wm_alt/' + sub + '_' + str(i) + '.png'
            imageio.imwrite(fname, wm[::-1,::-1,i].T.astype(np.uint8))
            
            # 255
            
            fname = tgt + 'vent/' + sub + '_' + str(i) + '.png'
            imageio.imwrite(fname, scale_to_255(ventri_all[::-1,::-1,i].T))
            
            fname = tgt + 'bet/' + sub + '_' + str(i) + '.png'
            imageio.imwrite(fname, scale_to_255(brain_data_bin[::-1,::-1,i].T))
            
            fname = tgt + 'tissue/' + sub + '_' + str(i) + '.png'
            imageio.imwrite(fname, scale_to_255(tissue[::-1,::-1,i].T))
            
            fname = tgt + 'csf/' + sub + '_' + str(i) + '.png'
            imageio.imwrite(fname, scale_to_255(csf[::-1,::-1,i].T))
            
            fname = tgt + 'gm/' + sub + '_' + str(i) + '.png'
            imageio.imwrite(fname, scale_to_255(gm[::-1,::-1,i].T))
            
            fname = tgt + 'wm/' + sub + '_' + str(i) + '.png'
            imageio.imwrite(fname, scale_to_255(wm[::-1,::-1,i].T))
            
            
            fname = tgt + 'ct_norm/' + sub + '_' + str(i) + '.tiff'
            imageio.imwrite(fname, ct_norm[::-1,::-1,i].T)
            
            fname = tgt + 'mri_norm/' + sub + '_' + str(i) + '.tiff'
            imageio.imwrite(fname, mri_norm[::-1,::-1,i].T)
            
            # data, no need to scale
            fname = tgt + 'ct/' + sub + '_' + str(i) + '.tiff'
            imageio.imwrite(fname, ct[::-1,::-1,i].T)
            
            fname = tgt + 'mri/' + sub + '_' + str(i) + '.tiff'
            imageio.imwrite(fname, mri[::-1,::-1,i].T)
            
#             nope no need, we can convert to uint8 in the code
#             fname = tgt + 'ct_int/' + sub + '_' + str(i) + '.tiff'
#             imageio.imwrite(fname, ct[::-1,::-1,i].T)
            
#             fname = tgt + 'mri_int/' + sub + '_' + str(i) + '.tiff'
#             imageio.imwrite(fname, mri[::-1,::-1,i].T)
            
    # break