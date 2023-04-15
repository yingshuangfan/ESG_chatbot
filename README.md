# ESG_chatbot
Group course work for ARIN7102 (HKU 2023)

## Pre-Processing

See more in the official document API:
https://pypdf2.readthedocs.io/en/stable/modules/PageObject.html#pypdf._page.PageObject.extract_text

## Summarization


## Prompting

See more about Question-Answering by hugging-face:
https://huggingface.co/docs/transformers/v4.27.2/en/tasks/question_answering#question-answering

SQuAD training data samples format:

```json
{"id": "5733be284776f41900661182", "title": "University_of_Notre_Dame", "context": "Architecturally, the school has a Catholic character. Atop the Main Building\"s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend \"Venite Ad Me Omnes\". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.", "question": "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?", "answers": {"text": ["Saint Bernadette Soubirous"], "answer_start": [515]}}
```

## Run API-Server with ES
```bash
cd elasticsearch
docker-compose up -d
# create index
python index.py <lang> <data_file> <index_name>
# start query server
python query.py 
```

