# export ONECCL_BINDINGS_FOR_PYTORCH_ENV_VERBOSE=1
# export CCL_ZE_IPC_EXCHANGE=sockets
export CCL_PROCESS_LAUNCHER=none 
# python -m torch.distributed.launch --nnodes 1 --nproc_per_node 8 finetuning_xpu.py --enable_fsdp --peft_method lora --dataset alpaca_dataset --model_name /media/newdrive2/huggingface/llama2-7b 2>&1 | tee test_raw_fsdp_peft_padding.log

export PROFILE=1
torchrun --nnodes 1 --nproc_per_node 2 finetuning_xpu.py --enable_fsdp --peft_method lora --dataset alpaca_dataset --model_name /media/newdrive2/huggingface/llama2-7b 2>&1 | tee finetune_fsdp_peft_padding_2rank_profile.log
