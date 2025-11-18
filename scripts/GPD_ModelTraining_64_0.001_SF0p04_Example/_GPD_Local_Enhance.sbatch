#!/bin/bash
#
#SBATCH --partition=ogsgpu_el9_tmp
#SBATCH --gres=gpu:1
#SBATCH --output=GPD_Local_Enhance_%J_.txt
#SBATCH --error=GPD_Local_Enhance_%J_.txt

#SBATCH --mem=20G
#SBATCH --time=48:00:00

#SBATCH --mail-user=hongyu.xiao-1@ou.edu      
#SBATCH --mail-type=BEGIN,END,FAIL

python GPD_Local_Enhance_Dual_RePlotExample_correctly_ver_3.py
