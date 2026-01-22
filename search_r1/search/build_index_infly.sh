corpus_file=/scratch/hc3337/qampari_searchr1/qampari_corpus.jsonl # jsonl
save_dir=/scratch/hc3337/qampari_searchr1/
retriever_name="infly_finetuned" # this is for indexing naming
retriever_model="/scratch/hc3337/models/iterative_retrieval/infly-finetuned/"

# change faiss_type to HNSW32/64/128 for ANN indexing
# change retriever_name to bm25 for BM25 indexing
CUDA_VISIBLE_DEVICES=0 python search_r1/search/index_builder.py \
    --retrieval_method $retriever_name \
    --model_path $retriever_model \
    --corpus_path $corpus_file \
    --save_dir $save_dir \
    --use_fp16 \
    --max_length 256 \
    --batch_size 512 \
    --pooling_method last \
    --faiss_type Flat \
    --save_embedding
