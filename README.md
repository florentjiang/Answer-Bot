# A Study on Question Answering Assessment with a ConversationalFlavour


To Launch the training and evaluation, use the following command:

````bash
python run_newsqa.py \
       --model_type bert \
       --model_name_or_path bert-base-cased \
       --do_train \
       --do_eval \
       --do_lower_case \
       --train_file NEWSQA/combined-newsqa-data-v1.json \
       --predict_file NEWSQA/combined-newsqa-data-v1.json \
       --per_gpu_train_batch_size 16 \
       --learning_rate 3e-5 \
       --num_train_epochs 2.0 \
       --max_seq_length 320 \
       --doc_stride 128 \
       --save_steps 10000 \
       --output_dir tmp_test_news_bert/
```

Dependencies:

```
pytorch
huggingface
Flask
```