echo "Validation during traning..."
method_=${1}

python3 sod_valid_sod.py   \
    --method  $method_ \
    --dataset  'ECSSD' 

python3 sod_valid_sod.py   \
    --method  $method_ \
    --dataset  'PASCAL-S' 


 
