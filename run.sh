for SEED in 0 1 2 3 4
do
  python main.py --policy $1 --env $2 --offline $3 --seed $SEED --min_known_training_steps $4 --alpha_unknown $5
done