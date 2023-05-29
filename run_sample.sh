SAMPLE_FLAGS="--batch_size 1 --num_samples 3000 --timestep_respacing 250"
MODEL_FLAGS="--channel_mult 1,2,2,4,4,4 --class_cond False --patch_size 4 --image_size 1024 --learn_sigma True --noise_schedule linear0.025 --num_channels 128 --num_head_channels 64 --num_res_blocks 2 --use_fp16 True --use_scale_shift_norm True --use_new_attention_order True"
python scripts/image_sample.py $MODEL_FLAGS --save_dir '/mnt/localssd/PatchDiffusionResults2' --model_path ./ffhq_weights.pt $SAMPLE_FLAGS 
