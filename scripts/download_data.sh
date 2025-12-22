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
       --eid 5c0c560e-9e1f-45e9-b66e-e4ee7855be84 \
       --base_path /media/lenny-aharon/T7/ibl-mouse/ibl-mouse_neural-activity

conda deactivate

cd ../scripts

# eids_test = [
#     '15b69921-d471-4ded-8814-2adad954bcd8',  # vertical bright strip right
#     '15763234-d21e-491f-a01b-1238eb96d389',  # dark
#     'aad23144-0e52-4eac-80c5-c4ee2decb198',  # wire by tongue
#     '9b528ad0-4599-4a55-9148-96cc1d93fb24',  # vertical bright band left
#     '5c0c560e-9e1f-45e9-b66e-e4ee7855be84',  # vertical bright band left, bright spot right, back paws
# ]