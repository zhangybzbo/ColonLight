python code/test.py --chunk /playpen2/COLON_RNNSLAM_TEST/sequences/000/image/ \
--model_name gamma_thru10_first2prev2_ssimALL \
--gpu 1 --num_adj=1 \
--seq_num=10 \
--exp_patch=32 \
--epochs=7 | tee adj.log
