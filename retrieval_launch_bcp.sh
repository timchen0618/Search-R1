
file_path=/scratch/hc3337/bcp_searchr1

retriever_name=qwen3-8b
index_file=$file_path/${retriever_name}_Flat.index
corpus_file=$file_path/browsecomp_plus_corpus.jsonl

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

python search_r1/search/retrieval_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 3 \
                                            --retriever_name $retriever_name \
                                            --retriever_model $retriever_model \
                                            --faiss_gpu
