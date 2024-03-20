# arr=( /data/data_repo/neuro_img/anat_brain_img/datasets/CERMEP-IDB-MRXFDG/processed/yihao/labels_new_test/ct_norm_sag/* ) 
arr=( /data/data_repo/neuro_img/anat_brain_img/datasets/CERMEP-IDB-MRXFDG/processed/yihao/labels_new_test/ct_norm/* ) 

python predict.py -i ${arr[@]} --model checkpoints_bet_alt/checkpoint_epoch5.pth --task bet_alt --gpu 1