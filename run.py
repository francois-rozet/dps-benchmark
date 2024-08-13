import os
import argparse
import yaml

from dawgz import job, schedule
from functools import partial


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def main(
    model_config: dict,
    diffusion_config: dict,
    task_config: dict,
    method_config: dict,
    basename: str,
    root: str,
    save: bool = True,
    seed: int = 0,
    gpu: int = 0,
):
    import numpy
    import torch

    from piqa import PSNR, SSIM, LPIPS
    from torchvision import transforms
    from torchvision.transforms.functional import to_pil_image

    from data.dataloader import get_dataset, get_dataloader
    from guided_diffusion.condition_methods import get_conditioning_method
    from guided_diffusion.measurements import get_noise, get_operator
    from guided_diffusion.unet import create_model
    from guided_diffusion.gaussian_diffusion import create_sampler
    from util.img_utils import mask_generator
    from util.logger import get_logger

    # Logger
    logger = get_logger()

    # RNG
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    logger.info(f"Random seed set to {seed}.")

    # Device setting
    device_str = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)

    # Load model
    model = create_model(**model_config)
    model = model.to(device)
    model.eval()

    # Prepare operator and noise
    measure_config = task_config["measurement"]
    operator = get_operator(device=device, **measure_config["operator"])
    noiser = get_noise(**measure_config["noise"])
    logger.info(f"Task: {measure_config['operator']['name']}")

    # Prepare conditioning method
    cond_method = get_conditioning_method(
        method_config["name"], operator, noiser, **method_config["params"]
    )
    measurement_cond_fn = cond_method.conditioning
    logger.info(f"Method: {method_config['name']}")

    ## In case of inpainting, we need to generate a mask
    if measure_config["operator"]["name"] == "inpainting":
        mask_gen = mask_generator(**measure_config["mask_opt"])

    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config)
    sample_fn = partial(
        sampler.p_sample_loop,
        model=model,
        measurement_cond_fn=measurement_cond_fn,
    )

    # Working directory
    os.makedirs(root, exist_ok=True)

    # Prepare dataloader
    data_config = task_config["data"]
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    dataset = get_dataset(**data_config, transforms=transform)
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)

    # Metrics
    metrics = {
        "PSNR": PSNR().to(device),
        "SSIM": SSIM().to(device),
        "LPIPS": LPIPS().to(device),
    }

    # Run inference
    for i, x in enumerate(loader):
        x = x.to(device)

        if measure_config["operator"]["name"] == "inpainting":
            mask = mask_gen(x)
            mask = mask[:, 0, :, :].unsqueeze(dim=0)
            measurement_cond_fn = partial(cond_method.conditioning, mask=mask)
            sample_fn = partial(sample_fn, measurement_cond_fn=measurement_cond_fn)

            y = noiser(operator.forward(x, mask=mask))
        else:
            y = noiser(operator.forward(x))

        ## Sample
        torch.cuda.reset_peak_memory_stats(device)

        x_start = torch.randn(x.shape, device=device).requires_grad_()

        with torch.random.fork_rng():
            x_pred = sample_fn(
                x_start=x_start,
                measurement=y,
                record=False,
                save_root=None,
            )

        logger.info(f"Memory: {torch.cuda.max_memory_allocated(device) / 2**30:.3f} GB")

        ## Eval
        x = torch.clip((x + 1) / 2, 0, 1)
        y = torch.clip((y + 1) / 2, 0, 1)
        x_pred = torch.clip((x_pred + 1) / 2, 0, 1)

        values = []

        for name, metric in metrics.items():
            value = metric(x, x_pred).item()
            values.append(value)
            logger.info(f"{name}: {value:.4f}")

        with open("metrics.csv", "a") as f:
            f.write(f"{basename},{seed},{i}," + ",".join(map(str, values)))
            f.write("\n")

        if save:
            img = to_pil_image(y.squeeze(0))
            img.save(os.path.join(root, f"{basename}_{i:03}_measurement.png"))

            img = to_pil_image(x_pred.squeeze(0))
            img.save(os.path.join(root, f"{basename}_{i:03}_prediction.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-config",
        type=str,
        default="./configs/model_ffhq_config.yaml",
    )
    parser.add_argument(
        "--diffusion-config",
        type=str,
        default="./configs/diffusion_config.yaml",
    )
    parser.add_argument(
        "--task-config",
        type=str,
        default="./configs/tasks/inpainting_random_config.yaml",
    )
    parser.add_argument("--method", type=str, default="mmps")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--maxiter", type=int, default=1)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--basename", type=str, default="image")
    parser.add_argument("--root", type=str, default="./results")
    parser.add_argument("--save", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--slurm", default=False, action="store_true")
    args = parser.parse_args()

    # Configs
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    diffusion_config["timestep_respacing"] = args.steps
    task_config = load_yaml(args.task_config)
    method_config = {
        "name": args.method,
        "params": {
            "scale": args.scale,
            "maxiter": args.maxiter,
        },
    }

    # Run
    f = partial(
        main,
        model_config=model_config,
        diffusion_config=diffusion_config,
        task_config=task_config,
        method_config=method_config,
        basename=args.basename,
        root=args.root,
        save=args.save,
        gpu=args.gpu,
        seed=args.seed,
    )

    if args.slurm:
        from dawgz import job, schedule

        schedule(
            job(
                f,
                name=args.basename,
                cpus=1,
                gpus=1,
                ram="16GB",
                time="06:00:00",
                partition="gpu",
            ),
            name=args.basename,
            backend="slurm",
            export="ALL",
            account="ariacpg",
        )
    else:
        f()
