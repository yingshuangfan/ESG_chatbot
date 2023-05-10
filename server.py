import time
import logging
import argparse
logging.basicConfig(level=logging.ERROR)

from elasticsearch import Elasticsearch

from es.query import Querier
# from utils.generator import inference
from utils.summarize import inference, inference_long

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--es_server", default="http://localhost:9200", type=str, required=True,
                        help="Elasticsearch service address")
    parser.add_argument("--es_index", default="test-index", type=str, required=True,
                        help="Elasticsearch index with stored dataset")
    parser.add_argument("--top_n", default=3, type=str, required=True,
                        help="Top-N KNN arguments will be selected")
    args = parser.parse_args()
    
    es_client = Elasticsearch(
        args.es_server,
        verify_certs=False,
    )
    querier = Querier(es_client, args.es_index, "v1_marco_de", "v1_marco_ce")

    top_n = args.top_n
    while True:
        query = input('Query: ')

        candidates = querier.search(query, topk=top_n)

        answers = querier.sort(query, candidates)[:top_n]

        concat_context = "\n".join([a['para'] for a in answers])

        print('Answer:')
        long_response = inference_long(concat_context)
        print(long_response)
    


if __name__ == '__main__':
    main() 
