WORK_DIR=/scratch/hc3337/projects/Search-R1
LOCAL_DIR=$WORK_DIR/data/multihop_training

## process multiple dataset search format train file
DATA=hotpotqa,2wikimultihopqa,musique
python $WORK_DIR/scripts/data_process/qa_search_train_merge.py --local_dir $LOCAL_DIR --data_sources $DATA

## process multiple dataset search format test file
DATA=hotpotqa,2wikimultihopqa,musique,bamboogle
python $WORK_DIR/scripts/data_process/qa_search_test_merge.py --local_dir $LOCAL_DIR --data_sources $DATA



LOCAL_DIR=$WORK_DIR/data/musique_training
## process multiple dataset search format train file
DATA=musique
python $WORK_DIR/scripts/data_process/qa_search_train_merge.py --local_dir $LOCAL_DIR --data_sources $DATA

## process multiple dataset search format test file
DATA=hotpotqa,2wikimultihopqa,musique,bamboogle
python $WORK_DIR/scripts/data_process/qa_search_test_merge.py --local_dir $LOCAL_DIR --data_sources $DATA