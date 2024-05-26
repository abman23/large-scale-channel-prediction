# Large-Scale Channel Prediction Systems

This repo is the official implementation of ["Large-Scale Channel Prediction System for ICASSP 2023 Pathloss Radio Map Prediction Challenge"](https://ieeexplore.ieee.org/document/10095257).

## Highlights
- Our system achieves an RMSE of 0.02569 on the provided RadioMap3Dseer dataset, and 0.0383 on the challenge test set, placing it in the **1st-rank** of the ML competition @ IEEE ICASSP 2023.
- **RMSE**: 0.02569
<!-- - Run-time: 57.90 [sec] -->
<div align="center">
<img src="fig-prediction.png" alt="map_USC" width="500"/> 
</div>

## Citation

```
@inproceedings{lee2023large,
  author={Lee, Ju-Hyung and Lee, Joohan and Lee, Seon-Ho and Molisch, Andreas F.},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={PMNet: Large-Scale Channel Prediction System}, 
  year={2023},
}
```

<!-- ## Directories
- network: includes network code(pmnet)
- outputs: containing the estimated radio maps
    - RadioMap3DSeer_Test: Estimated images using the validation subset we determined in the training dataset. -->

## Available checkpoints / dataset
<!--["Link"](https://drive.google.com/drive/folders/1Muep-_-zrY0cryF28eMmZXC_s3PSE2gY?usp=drive_link)-->
- [checkpoint](https://drive.google.com/file/d/1vEJT2ZA6J5OVfWs4q5tYFcqozkcT0siE/view?usp=sharing) 

- [dataset](https://drive.google.com/file/d/1YW3RyM9KYBe110CXC5aZJJ0MAIti65bY/view)


## Setup
Install packages using the following instruction.
```
pip install -r requirements.txt
```
- Python 3.8.15
- torch cuda

## Evaluation

### 1. Run sh file
```
sh eval.sh
```

### 2. Run python file with arguments

```
python eval.py \
    --input_dir './RadioMap3DSeer/png' \
    --gt_dir './RadioMap3DSeer/gain' \
    --first_index 631 \
    --last_index 700 \
    --pretrained_model './checkpoints/radiomapseer3d_pmnetV3_V2_model_0.00076.pt' \
    --network_type 'pmnet_v3' \
    --output_dir './outputs/RadioMap3DSeer_Test'
```

The RMSE and run-time will be printed in results.txt file and the terminal where you run.

### Arguments
- input_dir : The directory where folders of input images such as antennasWHeight, buildingsWHeight exist. (It should be {your_directory}/png)
- gt_dir : The directory where ground truth images exist. (It should be {your_directory}/gain)
- first_index : The first index of map indices. For example, when we have 0_0.png ~ 83_79.png images, it should be 0 which is the first index of maps.
- last_index : The last index of map indices. For example, in the above case, it should be 83 which is the last index of maps.
- pretrained_model : The path of pretrained model.
- network_type : The type of network. (pmnet_v3)
- output_dir : The directory where predicted images are saved. (Default: {current_directory}/outputs)

- Note: To use other datasets, one need to set arguments: input_dir, gt_dir, first_index, last_index

