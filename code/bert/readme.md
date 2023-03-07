### DATA PREPROCESSING
In the appropriate data folders run:
+ ``cat pl_pdb-ud-train.conllu | grep -v "^#" | cut -f 2,4 | tr '\t' ' ' > train.txt.tmp``
+ ``cat pl_pdb-ud-dev.conllu | grep -v "^#" | cut -f 2,4 | tr '\t' ' ' > dev.txt.tmp``
+ ``cat pl_pdb-ud-test.conllu | grep -v "^#" | cut -f 2,4 | tr '\t' ' ' > test.txt.tmp``

In order to get the XPOS tags instead of UPOS run:
+ ``cat pl_pdb-ud-train.conllu | grep -v "^#" | cut -f 2,5 | tr '\t' ' ' > train.txt.tmp``
+ ``cat pl_pdb-ud-dev.conllu | grep -v "^#" | cut -f 2,5 | tr '\t' ' ' > dev.txt.tmp``
+ ``cat pl_pdb-ud-test.conllu | grep -v "^#" | cut -f 2,5 | tr '\t' ' ' > test.txt.tmp``

Later also:
+ ``cat memoirs_3k_corrected.conllu | grep -v "^#" | cut -f 2,4 | tr '\t' ' ' > test.txt.tmp``
+ ``cat memoirs_3k_corrected.conllu | grep -v "^#" | cut -f 2,5 | tr '\t' ' ' > test.txt.tmp``

Move the created files into the folder with the code (this one)

Set:
+ ``export MAX_LENGTH=128``
+ ``export BERT_MODEL=dkleczek/bert-base-polish-uncased-v1``

Run:
+ ``python3 scripts/preprocess.py train.txt.tmp $BERT_MODEL $MAX_LENGTH > train.txt``
+ ``python3 scripts/preprocess.py dev.txt.tmp $BERT_MODEL $MAX_LENGTH > dev.txt``
+ ``python3 scripts/preprocess.py test.txt.tmp $BERT_MODEL $MAX_LENGTH > test.txt``

Later also:
+ ``python3 scripts/preprocess.py test.txt.tmp $BERT_MODEL $MAX_LENGTH > test.txt``

Run:
+ ``cat train.txt dev.txt test.txt | cut -d " " -f 2 | grep -v "^$"| sort | uniq > labels.txt``

Set up:
+ ``export OUTPUT_DIR=polUPOS-model``
+ ``export BATCH_SIZE=32``
+ ``export NUM_EPOCHS=3``
+ ``export SAVE_STEPS=500``
+ ``export SEED=1``

Run:
+ ``python3 run_ner.py --data_dir ./data_UPOS/ --task_type POS --labels ./labels.txt --model_name_or_path $BERT_MODEL --output_dir $OUTPUT_DIR --max_seq_length  $MAX_LENGTH --num_train_epochs $NUM_EPOCHS --per_device_train_batch_size $BATCH_SIZE --save_steps $SAVE_STEPS --seed $SEED --do_train --do_eval --do_predict``

For predicting:
+ ``export MODEL_PATH=./polUPOS-model/``
+ ``export CONFIG=./polUPOS-model/config.json``
+ ``python3 run_ner.py --data_dir ./hist_test_UPOS/ --labels ./labels.txt --model_name_or_path $MODEL_PATH --config_name $CONFIG --output_dir $OUTPUT_DIR --max_seq_length  $MAX_LENGTH --num_train_epochs $NUM_EPOCHS --per_device_train_batch_size $BATCH_SIZE --save_steps $SAVE_STEPS --seed $SEED --do_predict``

Also run:
+ ``export OUTPUT_DIR=polXPOS-model``
+ ``python3 run_ner.py --data_dir ./data_XPOS/ --task_type POS --labels ./labels.txt --model_name_or_path $BERT_MODEL --output_dir $OUTPUT_DIR --max_seq_length  $MAX_LENGTH --num_train_epochs $NUM_EPOCHS --per_device_train_batch_size $BATCH_SIZE --save_steps $SAVE_STEPS --seed $SEED --do_train --do_eval --do_predict``
