#!/bin/bash

scripts=(
  "python src/experiments/1_window_size.py --window_size 10"
  "python src/experiments/1_window_size.py --window_size 15"
  "python src/experiments/1_window_size.py --window_size 20"
  "python src/experiments/1_window_size.py --window_size 25"
  "python src/experiments/1_window_size.py --window_size 30"
  "python src/experiments/1_window_size.py --window_size 35"
  "python src/experiments/1_window_size.py --window_size 40"
  "python src/experiments/1_window_size.py --window_size 45"
  "python src/experiments/1_window_size.py --window_size 50"
  "python src/experiments/1_window_size.py --window_size 55"
  "python src/experiments/1_window_size.py --window_size 60"
  "python src/experiments/1_window_size.py --window_size 65"
  "python src/experiments/1_window_size.py --window_size 70"
  "python src/experiments/1_window_size.py --window_size 75"
  "python src/experiments/1_window_size.py --window_size 80"
  "python src/experiments/1_window_size.py --window_size 85"
  "python src/experiments/1_window_size.py --window_size 90"
  "python src/experiments/1_window_size.py --window_size 95"
  "python src/experiments/1_window_size.py --window_size 100"
)

for script in "${scripts[@]}"; do
  echo "Running: $script"
  eval "$script"
done

sudo shutdown -h now