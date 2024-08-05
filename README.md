# DPS Benchmark

This fork of the [official DPS repository](https://github.com/DPS2022/diffusion-posterior-sampling) is adapted to benchmark diffusion posterior sampling algorithms, including [DPS](https://arxiv.org/abs/2209.14687), [PiGDM](https://openreview.net/forum?id=9_gsMA8MRKQ), [TMPD](https://arxiv.org/abs/2310.06721) and [MMPS](https://arxiv.org/abs/2405.13712).

![cover-img](./figures/cover.jpg)

## Getting started

1. Clone the repository and its dependencies

    ```
    git clone https://github.com/francois-rozet/dps-benchmark
    cd dps-benchmark
    git clone https://github.com/VinAIResearch/blur-kernel-space-exploring bkse
    git clone https://github.com/LeviBorodenko/motionblur motionblur
    ```

2. Install the Python dependencies

    ```
    pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
    pip install -r requirements.txt
    ```

3. Download a checkpoint from [link](https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh?usp=sharing) and place it into `./models`

    ```
    mkdir models
    mv ffhq_10m.pt ./models
    ```

4. Run an experiment

    ```
    python run.py \
    --model-config ./configs/model_ffhq_config.yaml \
    --task-config ./configs/tasks/inpainting_random_config.yaml \
    --method dps --steps 1000 \
    --seed 42 \
    ```
