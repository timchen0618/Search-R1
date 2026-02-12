file_path=/scratch/hc3337/qampari_searchr1
retriever_name=$1
index_file=$file_path/${retriever_name}_Flat.index
corpus_file=$file_path/qampari_corpus.jsonl

if [ "$retriever_name" == "contriever" ]; then
    retriever_path="facebook/contriever-msmarco"
    pooling_method="mean"
elif [ "$retriever_name" == "infly" ]; then
    retriever_path="infly/inf-retriever-v1-1.5b"
    pooling_method="last"
elif [ "$retriever_name" == "qwen3-0.6b" ]; then
    retriever_path="Qwen/Qwen3-Embedding-0.6B"
    pooling_method="last"
elif [ "$retriever_name" == "contriever_finetuned" ]; then
    retriever_path="/scratch/hc3337/models/iterative_retrieval/contriever-finetuned/"
    pooling_method="mean"
elif [ "$retriever_name" == "infly_finetuned" ]; then
    retriever_path="/scratch/hc3337/models/iterative_retrieval/infly-finetuned/"
    pooling_method="last"
elif [ "$retriever_name" == "qwen3-0.6b_finetuned" ]; then
    retriever_path="/scratch/hc3337/models/iterative_retrieval/qwen3-finetuned/"
    pooling_method="last"
else
    echo "Invalid retriever name"
    exit 1
fi

echo "Retriever name: $retriever_name"
echo "Pooling method: $pooling_method"
python search_r1/search/retrieval_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 3 \
                                            --retriever_name $retriever_name \
                                            --retriever_model $retriever_path \
                                            --pooling_method $pooling_method \
					                        --faiss_gpu \
                                            --port $2
