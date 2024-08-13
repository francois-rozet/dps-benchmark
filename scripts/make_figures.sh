#!/usr/bin/bash

for task in inpainting_box inpainting_random motion_deblur super_resolution
do
    for steps in 0010 0100 1000
    do
        # DPS
        python run.py --task-config ./configs/tasks/${task}_config.yaml --method dps --scale 0.5 --steps $steps --basename ${task}_dps_${steps} --slurm &

        # PGDM
        python run.py --task-config ./configs/tasks/${task}_config.yaml --method pgdm --maxiter 5 --steps $steps --basename ${task}_pgdm_${steps} --slurm &

        # TMPD
        python run.py --task-config ./configs/tasks/${task}_config.yaml --method tmpd --maxiter 5 --steps $steps --basename ${task}_tmpd_${steps} --slurm &

        # MMPS
        for maxiter in 01 05
        do
            python run.py --task-config ./configs/tasks/${task}_config.yaml --method mmps --maxiter $maxiter --steps $steps --basename ${task}_mmps_${steps}_${maxiter} --slurm &
        done
    done
done

wait
