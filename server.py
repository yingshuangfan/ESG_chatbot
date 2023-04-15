import time

from elasticsearch import Elasticsearch

from es.query import Querier
# from generator import inference
from summarize import inference, inference_long

def main():
    
    es_client = Elasticsearch(
        "http://localhost:9200",
        verify_certs=False,
    )
    querier = Querier(es_client, "test-index", "v1_marco_de", "v1_marco_ce")

    top_n = 3
    while True:
        query = input('Query: ')

        start = time.time()
        candidates = querier.search(query)
        # print('Candidates:')
        # for c in candidates:
        #     print(c['title'], '\t', c['para'])

        answers = querier.sort(query, candidates)[:top_n]
        print('Answers:')
        for a in answers:
            print(a['title'], '\t', a['para'], '\t', a['score'])
        print(f'search take {time.time() - start}s')

        start = time.time()
        print('Generated Answer:')
        concat_context = "\n".join([a['para'] for a in answers])
        # response = inference(question=query, title=a['title'], context=concat_context)
        response = inference(concat_context)
        print(response)
        print(f'summary take {time.time() - start}s')

        start = time.time()
        print('Generated Long Answer:')
        long_response = inference_long(concat_context)
        print(long_response)
        print(f'long summary take {time.time() - start}s')
    


if __name__ == '__main__':
    main()
