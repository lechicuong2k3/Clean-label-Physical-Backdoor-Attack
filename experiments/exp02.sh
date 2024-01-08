# export OMP_NUM_THREADS=3
# torchrun --standalone --nnodes=1 --nproc-per-node=2 main_dist.py --trigger=real_beard --recipe=naive --poisonkey=1-0 --alpha=0.5 --beta=0.0 --scenario=finetuning  --threatmodel=clean-multi-source 
# torchrun --standalone --nnodes=1 --nproc-per-node=2 main_dist.py --trigger=sunglasses --recipe=naive --poisonkey=1-3 --alpha=0.5 --beta=0.0 --scenario=finetuning  --threatmodel=clean-multi-source 
# torchrun --standalone --nnodes=1 --nproc-per-node=2 main_dist.py --trigger=red_headband --recipe=naive --poisonkey=1-2 --alpha=0.5 --beta=0.0 --scenario=finetuning  --threatmodel=clean-multi-source 

python main.py --trigger=real_beard --recipe=naive --poisonkey=1-0 --alpha=0.0 --beta=0.1 --scenario=finetuning  --threatmodel=clean-multi-source 
python main.py --trigger=sunglasses --recipe=naive --poisonkey=1-3 --alpha=0.0 --beta=0.1 --scenario=finetuning  --threatmodel=clean-multi-source 
python main.py --trigger=red_headband --recipe=naive --poisonkey=1-4 --alpha=0.0 --beta=0.1 --scenario=finetuning  --threatmodel=clean-multi-source 