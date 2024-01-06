# Testing the connection between trigger and target class by using sunglass ad iteratively changing target and source

# Sleeper's Agent
python main.py --poisonkey=5-1 --recipe=gradient-matching --attackoptim=signAdam --trigger=sunglass --seed=12345 --deterministic --beta=0.0 
python main.py --poisonkey=1-5 --recipe=gradient-matching --attackoptim=signAdam --trigger=sunglass --seed=12345 --deterministic --beta=0.0 

# HTBA
python main.py --poisonkey=5-1 --recipe=hidden-trigger --attackoptim=PGD --trigger=sunglass --seed=12345 --deterministic --beta=0.0 
python main.py --poisonkey=1-5 --recipe=hidden-trigger --attackoptim=PGD --trigger=sunglass --seed=12345 --deterministic --beta=0.0 