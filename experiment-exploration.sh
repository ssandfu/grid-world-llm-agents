#!/bin/bash
set -u
for i in "$@"; do
  case $i in
    -m=*|--model=*)
      model_name="${i#*=}"
      shift # past argument=value
      ;;
    -*|--*)
      echo "Unknown option $i"
      exit 1
      ;;
    *)
      ;;
  esac
done

#model_name="deepseek-r1"
runs_p_d=10 
declare -a arr=("easy" "medium" "hard")
for gridworld in "${arr[@]}"
do
    for i in $(seq 1 $runs_p_d)
    do
        MODEL_ID="$model_name" GRIDWORLD="$gridworld" python ollama-agent.py
    done
done