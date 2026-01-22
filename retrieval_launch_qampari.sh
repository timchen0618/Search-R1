file_path=/scratch/hc3337/qampari_searchr1
index_file=$file_path/contriever_Flat.index
corpus_file=$file_path/qampari_corpus.jsonl
retriever_name=contriever
retriever_path=facebook/contriever-msmarco

python search_r1/search/retrieval_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 3 \
                                            --retriever_name $retriever_name \
                                            --retriever_model $retriever_path \
                                            --pooling_method mean \
					    --faiss_gpu
