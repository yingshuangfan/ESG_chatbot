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
    args = parser.parse_args()

    companies = []
    for root, dir, files in os.walk(args.data_dir):
        for file in files:
            if file.endswith(".tsv"):
                company_name = parse_company_name(file_name=file)
                companies.append(company_name)
    print(companies)

    es_client = Elasticsearch(
        "http://localhost:9200",
        verify_certs=False,
    )
    querier = Querier(es_client, "test-index-v3", "v1_marco_de", "v1_marco_ce")
    top_n = 3

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
