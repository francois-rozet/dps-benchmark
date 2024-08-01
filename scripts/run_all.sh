#!/usr/bin/bash

for task in inpainting_random inpainting_box motion_deblur super_resolution
do
    for steps in 0010 0100 1000
    do
        # DPS
        python sample_condition.py --task_config ./configs/tasks/${task}_config.yaml --method ps --scale 0.5 --steps $steps --basename ${task}_dps_${steps} --slurm &

        # PGDM
        for maxiter in 01 16
        do
            python sample_condition.py --task_config ./configs/tasks/${task}_config.yaml --method pgdm --maxiter $maxiter --steps $steps --basename ${task}_pgdm_${steps}_${maxiter} --slurm &
        done

        # MMPS
        for maxiter in 01 05
        do
            python sample_condition.py --task_config ./configs/tasks/${task}_config.yaml --method mmps --maxiter $maxiter --steps $steps --basename ${task}_mmps_${steps}_${maxiter} --slurm &
        done
    done
done

wait
