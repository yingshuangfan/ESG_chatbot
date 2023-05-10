import time
import logging
logging.basicConfig(level=logging.ERROR)

from elasticsearch import Elasticsearch

from es.query import Querier
# from utils.generator import inference
from utils.summarize import inference, inference_long

def main():
    
    es_client = Elasticsearch(
        "http://localhost:9200",
        verify_certs=False,
    )
    querier = Querier(es_client, "test-index-v3", "v1_marco_de", "v1_marco_ce")

    top_n = 3
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
