# export ONECCL_BINDINGS_FOR_PYTORCH_ENV_VERBOSE=1
# export CCL_ZE_IPC_EXCHANGE=sockets
export CCL_PROCESS_LAUNCHER=none 
# python -m torch.distributed.launch --nnodes 1 --nproc_per_node 8 finetuning_xpu.py --enable_fsdp --peft_method lora --dataset alpaca_dataset --model_name /media/newdrive2/huggingface/llama2-7b 2>&1 | tee test_raw_fsdp_peft_padding.log

# export PROFILE=1
# torchrun --nnodes 1 --nproc_per_node 2 finetuning_xpu.py --enable_fsdp --peft_method lora --dataset alpaca_dataset --model_name /media/newdrive2/huggingface/llama2-7b 2>&1 | tee finetune_fsdp_peft_padding_2rank_verbose.log


# fp16
# torchrun --nnodes 1 --nproc_per_node 2 finetuning_xpu.py --enable_fsdp --peft_method lora --dataset alpaca_dataset  --use_fp16 --model_name /media/newdrive2/huggingface/llama2-7b 2>&1 | tee finetune_fsdp_peft_2rank_fp16.log


## bf16
export PROFILE=1
torchrun --nnodes 1 --nproc_per_node 2 finetuning_xpu.py --enable_fsdp --peft_method lora --dataset alpaca_dataset --pure_bf16 --model_name /media/newdrive2/huggingface/llama2-7b 2>&1 | tee finetune_fsdp_peft_2rank_bf16_profile.log
