torchrun --standalone --nnodes=1 --rdzv-endpoint=localhost:29450 --nproc-per-node=2 main_dist.py --trigger=real_beard --recipe=gradient-matching --poisonkey=3-0 --alpha=0.1 --beta=0.0 --scenario=finetuning --train_max_epoch=20 --threatmodel=clean-single-source 
torchrun --standalone --nnodes=1 --rdzv-endpoint=localhost:29450 --nproc-per-node=2 main_dist.py --trigger=real_beard --recipe=gradient-matching --poisonkey=5-0 --alpha=0.1 --beta=0.0 --scenario=finetuning --train_max_epoch=20 --threatmodel=clean-single-source 
torchrun --standalone --nnodes=1 --rdzv-endpoint=localhost:29450 --nproc-per-node=2 main_dist.py --trigger=real_beard --recipe=gradient-matching --poisonkey=7-0 --alpha=0.1 --beta=0.0 --scenario=finetuning --train_max_epoch=20 --threatmodel=clean-single-source 

torchrun --standalone --nnodes=1 --rdzv-endpoint=localhost:29450 --nproc-per-node=2 main_dist.py --trigger=real_beard --recipe=gradient-matching --poisonkey=1-3 --alpha=0.1 --beta=0.0 --scenario=finetuning --train_max_epoch=20 --threatmodel=clean-single-source 
torchrun --standalone --nnodes=1 --rdzv-endpoint=localhost:29450 --nproc-per-node=2 main_dist.py --trigger=real_beard --recipe=gradient-matching --poisonkey=4-3 --alpha=0.1 --beta=0.0 --scenario=finetuning --train_max_epoch=20 --threatmodel=clean-single-source 
torchrun --standalone --nnodes=1 --rdzv-endpoint=localhost:29450 --nproc-per-node=2 main_dist.py --trigger=real_beard --recipe=gradient-matching --poisonkey=8-3 --alpha=0.1 --beta=0.0 --scenario=finetuning --train_max_epoch=20 --threatmodel=clean-single-source 

torchrun --standalone --nnodes=1 --rdzv-endpoint=localhost:29450 --nproc-per-node=2 main_dist.py --trigger=real_beard --recipe=gradient-matching --poisonkey=0-2 --alpha=0.1 --beta=0.0 --scenario=finetuning --train_max_epoch=20 --threatmodel=clean-single-source 
torchrun --standalone --nnodes=1 --rdzv-endpoint=localhost:29450 --nproc-per-node=2 main_dist.py --trigger=real_beard --recipe=gradient-matching --poisonkey=6-2 --alpha=0.1 --beta=0.0 --scenario=finetuning --train_max_epoch=20 --threatmodel=clean-single-source 
torchrun --standalone --nnodes=1 --rdzv-endpoint=localhost:29450 --nproc-per-node=2 main_dist.py --trigger=real_beard --recipe=gradient-matching --poisonkey=9-2 --alpha=0.1 --beta=0.0 --scenario=finetuning --train_max_epoch=20 --threatmodel=clean-single-source 