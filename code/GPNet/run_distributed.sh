#!/bin/bash
NNODES=2
# Local command
module load cuda-toolkit/11.8.0
python -m torch.distributed.launch --nproc_per_node=1 --nnodes=$NNODES --node_rank=0 --master_addr="172.21.7.53" --master_port=1234 main.py --local-rank=0 &

sleep 2s

# Remote command via SSH
ssh haolu@172.21.7.54 '
module load cuda-toolkit/11.8.0
conda activate rank
cd /isilon/datalake/cialab/scratch/cialab/Hao/work_record/Project1_GM/codes/Point_cloud_gene_expression
python -m torch.distributed.launch --nproc_per_node=1 --nnodes=$NNODES --node_rank=1 --master_addr="172.21.7.53" --master_port=1234 main.py --local-rank=0 &
'

# # Remote command via SSH
# ssh haolu@172.21.7.52 '
# module load cuda-toolkit/11.8.0
# conda activate rank
# cd /isilon/datalake/cialab/scratch/cialab/Hao/work_record/Project1_GM/codes/Point_cloud_gene_expression
# python -m torch.distributed.launch --nproc_per_node=1 --nnodes=$NNODES --node_rank=2 --master_addr="172.21.7.53" --master_port=1234 main.py --local-rank=0 &
# '
# # Remote command via SSH
# ssh haolu@172.21.7.51 '
# module load cuda-toolkit/11.8.0
# conda activate rank
# cd /isilon/datalake/cialab/scratch/cialab/Hao/work_record/Project1_GM/codes/Point_cloud_gene_expression
# python -m torch.distributed.launch --nproc_per_node=1 --nnodes=$NNODES --node_rank=3 --master_addr="172.21.7.53" --master_port=1234 main.py --local-rank=0 &
# '

# Wait for all background processes to finish
wait