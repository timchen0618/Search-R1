
corpus_file=/scratch/hc3337/bcp_searchr1/browsecomp_plus_corpus.jsonl # jsonl
save_dir=/scratch/hc3337/bcp_searchr1/
retriever_name="qwen3-8b" # this is for indexing naming

if [ "$retriever_name" == "qwen3-0.6b" ]; then
    retriever_model="Qwen/Qwen3-Embedding-0.6B"
elif [ "$retriever_name" == "qwen3-8b" ]; then
    retriever_model="Qwen/Qwen3-Embedding-8B"
elif [ "$retriever_name" == "infly" ]; then
    retriever_model="infly/inf-retriever-v1-1.5b"
elif [ "$retriever_name" == "contriever" ]; then
    retriever_model="facebook/contriever-msmarco"
else
    echo "Invalid retriever name"
    exit 1
fi

# change faiss_type to HNSW32/64/128 for ANN indexing
# change retriever_name to bm25 for BM25 indexing
CUDA_VISIBLE_DEVICES=0 python search_r1/search/index_builder.py \
    --retrieval_method $retriever_name \
    --model_path $retriever_model \
    --corpus_path $corpus_file \
    --save_dir $save_dir \
    --use_fp16 \
    --max_length 512 \
    --batch_size 512 \
    --pooling_method last \
    --faiss_type Flat \
    --save_embedding \
