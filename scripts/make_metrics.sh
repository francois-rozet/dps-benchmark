#!/usr/bin/bash

for task in inpainting_box inpainting_random motion_deblur super_resolution
do
    for steps in 10 100 1000
    do
        # DPS
        python run.py --task-config ./configs/tasks/${task}_config.yaml --method dps --scale 0.5 --steps $steps --basename ${task}_dps_${steps} --slurm --no-save &

        # PGDM
        python run.py --task-config ./configs/tasks/${task}_config.yaml --method pgdm --maxiter 5 --steps $steps --basename ${task}_pgdm_${steps} --slurm --no-save &

        # TMPD
        python run.py --task-config ./configs/tasks/${task}_config.yaml --method tmpd --maxiter 5 --steps $steps --basename ${task}_tmpd_${steps} --slurm --no-save &

        # DiffPIR
        python run.py --task-config ./configs/tasks/${task}_config.yaml --method diffpir --maxiter 5 --steps $steps --basename ${task}_diffpir_${steps} --slurm --no-save &

        # MMPS
        for maxiter in 1 5
        do
            python run.py --task-config ./configs/tasks/${task}_config.yaml --method mmps --maxiter $maxiter --steps $steps --basename ${task}_mmps_${steps}_${maxiter} --slurm --no-save &
        done
    done
done

wait
