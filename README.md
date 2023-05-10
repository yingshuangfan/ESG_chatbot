# ESG ConsultBot
Group course work for ARIN7102 (HKU 2023)

## Create Environment

```bash
conda env create -f environment.yml
```

## Run API-Server with ES

1. run preprocessing
```bash
python preprocess.py --data_dir ./data/PDF --output_dir ./data/TXT --to_es true
```

2. start an elasticsearch single-node (with Docker)
```bash
cd elasticsearch
docker-compose up -d
```

3. create an index mapping, then start word embedding and the build local database
```bash
curl -X PUT -H "Content-Type: application/json" -d @mappings.json http://localhost:9200/test-index
python index.py en ../data/HSBC_HK_ESG_2022.tsv test-index
```

4. all set! start the qa-cli(server)
```bash
cd ..
python server.py 
```

## Example Results

**Q:** What did Adidas do in greenhouse gas emissions?

**A:** Adidas' estimated environmental impact for 2021 is split equally among the various value chain segments. For example, sourcing and processing are both significant sources of greenhouse gas (GHG) emissions; however, raw materials management is a significant source of GHG. In order to effectively manage these multiple sources of CO2s, Adidas plans to invest roughly 90% of its annual revenue in "sustainable processes."

## Create GPU environment in GPU-farm

```bash
conda create -n esg-chatbot python=3.8
conda activate esg-chatbot

pip install paddlepaddle-gpu
```
