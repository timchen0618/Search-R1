WORK_DIR=/scratch/hc3337/projects/Search-R1

## process multiple dataset search format train file
LOCAL_DIR=$WORK_DIR/data/bcp_train_test_split
python $WORK_DIR/scripts/data_process/bcp_search.py --local_dir $LOCAL_DIR --template_type bcp --train_test_split
python $WORK_DIR/scripts/data_process/bcp_search.py --local_dir $LOCAL_DIR --template_type base --train_test_split


## process multiple dataset search format test file
LOCAL_DIR=$WORK_DIR/data/bcp_test
python $WORK_DIR/scripts/data_process/bcp_search.py --local_dir $LOCAL_DIR --template_type bcp
python $WORK_DIR/scripts/data_process/bcp_search.py --local_dir $LOCAL_DIR --template_type base
