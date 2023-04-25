# -*- coding: utf-8 -*-

import argparse
import os
import sys

# import faiss
import tqdm
import numpy as np
import rocketqa
from elasticsearch import Elasticsearch, helpers


class Indexer:
    def __init__(self, es_client, index_name, model):
        self.es_client = es_client
        self.index_name = index_name
        self.dual_encoder = rocketqa.load_model(
            model=model,
            use_cuda=False, # GPU: True
            device_id=0,
            batch_size=32,
        )

    def index(self, tps):
        titles, paras = zip(*tps)
        embs = self.dual_encoder.encode_para(para=paras, title=titles)
    
        def gen_actions():
            for i, emb in enumerate(embs):
                # Normalize the NumPy array to a unit vector to use `dot_product` similarity,
                # see https://www.elastic.co/guide/en/elasticsearch/reference/current/dense-vector.html#dense-vector-params.
                emb = emb / np.linalg.norm(emb)
                yield dict(
                    _index=self.index_name,
                    # _id=i+1,
                    _source=dict(
                        title=titles[i],
                        paragraph=paras[i],
                        vector=emb,
                    ),
                )
        return helpers.bulk(self.es_client, gen_actions())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../data/TXT", type=str, required=True,
                        help="parsed files(in TXT format)")
    parser.add_argument('--lang', choices=['zh', 'en'], default="en", help='The language', required=False)
    # parser.add_argument('--data_file', help='The data file', required=False)
    parser.add_argument('--index_name', help='The index name', required=True)
    args = parser.parse_args()

    # if args.lang == 'zh':
    #     model = 'zh_dureader_de_v2'
    # elif args.lang == 'en':
    #     model = 'v1_marco_de'
    model = "v1_marco_de"
    es_client = Elasticsearch(
        "http://localhost:9200",
        verify_certs=False,
    )
    indexer = Indexer(es_client, args.index_name, model)

    paths = [os.path.join(root, file) for root, dir, files in os.walk(args.data_dir) 
             for file in files if file.endswith(".tsv")]
    for path in tqdm.tqdm(paths):
        # check if company has been loaded
        with open(path, "r") as fp:
            line = fp.readline()
            company_name = line.split("\t")[0]
            if not company_name:
                print(f"vacant file: {path}")
                continue
            result = es_client.search(index=args.index_name, query={"match":{"title":{"query":company_name}}})
            total = result["hits"]["total"]["value"]
            if total > 0:
                print(f"skip {company_name}, because it's already indexed")
            
            print(f"indexing {company_name}...")
            try:
                tps = [line.strip().split('\t') for line in fp]
                result = indexer.index(tps)
                print(path, result)
            except Exception as err:
                print(f"error in indexing: {err}")


if __name__ == '__main__':
    main()
