isp='random'
method='speculative_jacobi'
num_init_new_token=16
benchmark_way='order'
prompt="MSCOCO2017Val"
num_images=4000
slice='1759-4000'
output_path=
mkdir -p ${output_path}

nohup python ../main.py \
  --output_path=$output_path \
  --isp=$isp \
  --method=$method \
  --num_init_new_token=$num_init_new_token \
  --benchmark_way=$benchmark_way \
  --prompt=$prompt \
  --num_images=$num_images \
  --slice=$slice \
  --static_tree \
  --tree_choices=Grouped_Tree_2 > ${output_path}.log 2>&1 &