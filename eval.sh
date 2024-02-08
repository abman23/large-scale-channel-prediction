python eval.py \
    --input_dir './RadioMap3DSeer/png' \
    --gt_dir './RadioMap3DSeer/gain' \
    --first_index 631 \
    --last_index 700 \
    --pretrained_model './checkpoints/radiomapseer3d_pmnetV3_V2_model_0.00076.pt' \
    --network_type 'pmnet_v3' \
    --output_dir './outputs/RadioMap3DSeer_Test'