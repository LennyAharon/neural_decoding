#!/bin/bash
#SBATCH --account=stats             
#SBATCH --job-name="data"
#SBATCH --output="data.%j.out"
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1        
#SBATCH --mem-per-cpu=20G       
#SBATCH --time=0-1:00              

# module load anaconda

# . ~/.bashrc

eid=${1}

echo $TMPDIR

conda activate decoding

cd /home/lenny-aharon/neural_decoding

python src/download_data.py \
       --eid 9b528ad0-4599-4a55-9148-96cc1d93fb24 \
       --base_path /media/lenny-aharon/T7/ibl-mouse/ibl-mouse_neural-activity

conda deactivate

cd ../scripts
