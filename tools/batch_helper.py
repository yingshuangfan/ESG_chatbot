"""
Helper to Generate QA-pairs on Test dataset with the question template.
"""

import os
import tqdm
import argparse

from preprocess import parse_company_name
from es.query import Querier
from elasticsearch import Elasticsearch
from utils.summarize import inference, inference_long

questions = [
    "greenhouse gas emissions", "waste and pollution", "water use", "land use and biodiversity", 
    "workforce and diversity", "safety management", "customer engagement", "communities", 
    "structure and oversight", "code and values", "transparency and reporting", "financial and operational risks"
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/", type=str, required=True,
                        help="original raw files(in PDF format)")
    parser.add_argument("--es_server", default="http://localhost:9200", type=str, required=True,
                        help="Elasticsearch service address")
    parser.add_argument("--es_index", default="test-index", type=str, required=True,
                        help="Elasticsearch index with stored dataset")
    parser.add_argument("--top_n", default=3, type=str, required=True,
                        help="Top-N KNN arguments will be selected")
    args = parser.parse_args()

    companies = []
    for root, dir, files in os.walk(args.data_dir):
        for file in files:
            if file.endswith(".tsv"):
                company_name = parse_company_name(file_name=file)
                companies.append(company_name)
    print(companies)

    es_client = Elasticsearch(
        args.es_server,
        verify_certs=False,
    )
    querier = Querier(es_client, args.es_index, "v1_marco_de", "v1_marco_ce")
    top_n = args.top_n

    for company in companies:
        outputs = []
        print(f"QA for {company}")
        for keyword in tqdm.tqdm(questions):
            try:
                query = f"What did {company} do in {keyword}?"
                outputs.append("Q: " + query + "\n")

                candidates = querier.search(query, title=company, topk=top_n)
                if not candidates:
                    outputs.append("A: N/A\n")
                    continue

                answers = querier.sort(query, candidates)[:top_n]

                concat_context = "\n".join([a['para'] for a in answers])
                # response = inference(concat_context)
                long_response = inference_long(concat_context)

                outputs.append("A: " + long_response + "\n")

            except Exception as err:
                print(f"skip {company} for {err}")

        with open(f"output/test/{company}_QA.txt", "w") as fp:
            fp.writelines(outputs)     


if __name__ == "__main__":
    main()
