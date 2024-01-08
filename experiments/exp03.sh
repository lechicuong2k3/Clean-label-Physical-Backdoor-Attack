# export OMP_NUM_THREADS=3
# torchrun --standalone --nnodes=1 --rdzv-endpoint=localhost:29450 --nproc-per-node=2 main_dist.py --trigger=sunglasses --recipe=gradient-matching --poisonkey=1-3 --alpha=0.5 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source 
# torchrun --standalone --nnodes=1 --rdzv-endpoint=localhost:29450 --nproc-per-node=2 main_dist.py --trigger=sunglasses --recipe=gradient-matching --poisonkey=4-3 --alpha=0.5 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source 
# torchrun --standalone --nnodes=1 --rdzv-endpoint=localhost:29450 --nproc-per-node=2 main_dist.py --trigger=sunglasses --recipe=gradient-matching --poisonkey=8-3 --alpha=0.5 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source 

# torchrun --standalone --nnodes=1 --rdzv-endpoint=localhost:29450 --nproc-per-node=2 main_dist.py --trigger=real_beard --recipe=gradient-matching --poisonkey=3-0 --alpha=0.5 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source 
# torchrun --standalone --nnodes=1 --rdzv-endpoint=localhost:29450 --nproc-per-node=2 main_dist.py --trigger=real_beard --recipe=gradient-matching --poisonkey=5-0 --alpha=0.5 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source 
# torchrun --standalone --nnodes=1 --rdzv-endpoint=localhost:29450 --nproc-per-node=2 main_dist.py --trigger=real_beard --recipe=gradient-matching --poisonkey=7-0 --alpha=0.5 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source 

# torchrun --standalone --nnodes=1 --rdzv-endpoint=localhost:29450 --nproc-per-node=2 main_dist.py --trigger=red_headband --recipe=gradient-matching --poisonkey=0-2 --alpha=0.5 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source 
# torchrun --standalone --nnodes=1 --rdzv-endpoint=localhost:29450 --nproc-per-node=2 main_dist.py --trigger=red_headband --recipe=gradient-matching --poisonkey=5-2 --alpha=0.5 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source 
# torchrun --standalone --nnodes=1 --rdzv-endpoint=localhost:29450 --nproc-per-node=2 main_dist.py --trigger=red_headband --recipe=gradient-matching --poisonkey=8-2 --alpha=0.5 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source 

python main.py --trigger=sunglasses --recipe=gradient-matching --poisonkey=1-3 --alpha=0.1 --beta=0.0 --scenario=finetuning --threatmodel=clean-single-source 
python main.py --trigger=sunglasses --recipe=gradient-matching --poisonkey=4-3 --alpha=0.1 --beta=0.0 --scenario=finetuning --threatmodel=clean-single-source 
python main.py --trigger=sunglasses --recipe=gradient-matching --poisonkey=8-3 --alpha=0.1 --beta=0.0 --scenario=finetuning --threatmodel=clean-single-source 

python main.py --trigger=real_beard --recipe=gradient-matching --poisonkey=3-0 --alpha=0.1 --beta=0.0 --scenario=finetuning --threatmodel=clean-single-source 
python main.py --trigger=real_beard --recipe=gradient-matching --poisonkey=5-0 --alpha=0.1 --beta=0.0 --scenario=finetuning --threatmodel=clean-single-source 
python main.py --trigger=real_beard --recipe=gradient-matching --poisonkey=7-0 --alpha=0.1 --beta=0.0 --scenario=finetuning --threatmodel=clean-single-source 

python main.py --trigger=red_headband --recipe=gradient-matching --poisonkey=0-4 --alpha=0.1 --beta=0.0 --scenario=finetuning --threatmodel=clean-single-source 
python main.py --trigger=red_headband --recipe=gradient-matching --poisonkey=5-4 --alpha=0.1 --beta=0.0 --scenario=finetuning --threatmodel=clean-single-source 
python main.py --trigger=red_headband --recipe=gradient-matching --poisonkey=9-4 --alpha=0.1 --beta=0.0 --scenario=finetuning --threatmodel=clean-single-source 