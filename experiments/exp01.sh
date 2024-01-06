export OMP_NUM_THREADS=3;
torchrun --standalone --nnodes=1 --nproc-per-node=3 main_dist.py --trigger=real_beard --recipe=naive --poisonkey=3-0 --alpha=0.0 --beta=0.1 --scenario=finetuning --train_max_epoch=20 --threatmodel=clean-multi-source 
torchrun --standalone --nnodes=1 --nproc-per-node=3 main_dist.py --trigger=real_beard --recipe=naive --poisonkey=5-0 --alpha=0.0 --beta=0.1 --scenario=finetuning --train_max_epoch=20 --threatmodel=clean-multi-source 
torchrun --standalone --nnodes=1 --nproc-per-node=3 main_dist.py --trigger=real_beard --recipe=naive --poisonkey=7-0 --alpha=0.0 --beta=0.1 --scenario=finetuning --train_max_epoch=20 --threatmodel=clean-multi-source 

torchrun --standalone --nnodes=1 --nproc-per-node=3 main_dist.py --trigger=real_beard --recipe=naive --poisonkey=1-3 --alpha=0.1 --beta=0.0 --scenario=finetuning --train_max_epoch=20 --threatmodel=clean-multi-source 
torchrun --standalone --nnodes=1 --nproc-per-node=3 main_dist.py --trigger=real_beard --recipe=naive --poisonkey=4-3 --alpha=0.1 --beta=0.0 --scenario=finetuning --train_max_epoch=20 --threatmodel=clean-multi-source 
torchrun --standalone --nnodes=1 --nproc-per-node=3 main_dist.py --trigger=real_beard --recipe=naive --poisonkey=8-3 --alpha=0.1 --beta=0.0 --scenario=finetuning --train_max_epoch=20 --threatmodel=clean-multi-source 

torchrun --standalone --nnodes=1 --nproc-per-node=3 main_dist.py --trigger=real_beard --recipe=naive --poisonkey=0-2 --alpha=0.1 --beta=0.0 --scenario=finetuning --train_max_epoch=20 --threatmodel=clean-multi-source 
torchrun --standalone --nnodes=1 --nproc-per-node=3 main_dist.py --trigger=real_beard --recipe=naive --poisonkey=6-2 --alpha=0.1 --beta=0.0 --scenario=finetuning --train_max_epoch=20 --threatmodel=clean-multi-source 
torchrun --standalone --nnodes=1 --nproc-per-node=3 main_dist.py --trigger=real_beard --recipe=naive --poisonkey=9-2 --alpha=0.1 --beta=0.0 --scenario=finetuning --train_max_epoch=20 --threatmodel=clean-multi-source 