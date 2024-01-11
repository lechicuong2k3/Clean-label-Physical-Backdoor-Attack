# export OMP_NUM_THREADS=3
# torchrun --standalone --rdzv-endpoint=localhost:21212 --nnodes=1 --nproc-per-node=2 main_dist.py --trigger=sunglasses --recipe=hidden-trigger --poisonkey=1-3 --alpha=0.5 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source 
# torchrun --standalone --rdzv-endpoint=localhost:21212 --nnodes=1 --nproc-per-node=2 main_dist.py --trigger=sunglasses --recipe=hidden-trigger --poisonkey=4-3 --alpha=0.5 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source 
# torchrun --standalone --rdzv-endpoint=localhost:21212 --nnodes=1 --nproc-per-node=2 main_dist.py --trigger=sunglasses --recipe=hidden-trigger --poisonkey=8-3 --alpha=0.5 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source 

# torchrun --standalone --rdzv-endpoint=localhost:21212 --nnodes=1 --nproc-per-node=2 main_dist.py --trigger=real_beard --recipe=hidden-trigger --poisonkey=3-0 --alpha=0.5 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source 
# torchrun --standalone --rdzv-endpoint=localhost:21212 --nnodes=1 --nproc-per-node=2 main_dist.py --trigger=real_beard --recipe=hidden-trigger --poisonkey=5-0 --alpha=0.5 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source 
# torchrun --standalone --rdzv-endpoint=localhost:21212 --nnodes=1 --nproc-per-node=2 main_dist.py --trigger=real_beard --recipe=hidden-trigger --poisonkey=7-0 --alpha=0.5 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source 

# torchrun --standalone --rdzv-endpoint=localhost:21212 --nnodes=1 --nproc-per-node=2 main_dist.py --trigger=red_headband --recipe=hidden-trigger --poisonkey=0-4 --alpha=0.5 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source 
# torchrun --standalone --rdzv-endpoint=localhost:21212 --nnodes=1 --nproc-per-node=2 main_dist.py --trigger=red_headband --recipe=hidden-trigger --poisonkey=5-4 --alpha=0.5 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source 
# torchrun --standalone --rdzv-endpoint=localhost:21212 --nnodes=1 --nproc-per-node=2 main_dist.py --trigger=red_headband --recipe=hidden-trigger --poisonkey=9-4 --alpha=0.5 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source 

python main.py --trigger=sunglasses --recipe=hidden-trigger --poisonkey=1-3 --alpha=0.1 --beta=0.0 --scenario=finetuning --threatmodel=clean-single-source 
python main.py --trigger=sunglasses --recipe=hidden-trigger --poisonkey=4-3 --alpha=0.1 --beta=0.0 --scenario=finetuning --threatmodel=clean-single-source 
python main.py --trigger=sunglasses --recipe=hidden-trigger --poisonkey=8-3 --alpha=0.1 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source 

python main.py --trigger=real_beard --recipe=hidden-trigger --poisonkey=3-0 --alpha=0.1 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source 
python main.py --trigger=real_beard --recipe=hidden-trigger --poisonkey=5-0 --alpha=0.1 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source 
python main.py --trigger=real_beard --recipe=hidden-trigger --poisonkey=7-0 --alpha=0.1 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source 

python main.py --trigger=red_headband --recipe=hidden-trigger --poisonkey=0-4 --alpha=0.1 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source 
python main.py --trigger=red_headband --recipe=hidden-trigger --poisonkey=5-4 --alpha=0.1 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source 
python main.py --trigger=red_headband --recipe=hidden-trigger --poisonkey=9-4 --alpha=0.1 --beta=0.0 --scenario=finetuning  --threatmodel=clean-single-source 