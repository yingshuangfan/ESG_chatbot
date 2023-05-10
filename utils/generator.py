"""
Retrieval-Augmented Generation with T5
reference: https://lilianweng.github.io/posts/2020-10-29-odqa/#open-book-qa-retriever-generator
"""

# from transformers import pipeline
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, T5ForConditionalGeneration

# checkpoint = "t5-base" (size:892M)
# reference: https://huggingface.co/docs/transformers/v4.28.1/en/model_doc/t5#transformers.T5ForConditionalGeneration

checkpoint = "t5-base"  
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = T5ForConditionalGeneration.from_pretrained(checkpoint)

def inference(question, title, context):
    input_ids = tokenizer(
        # f"question: {question}, title: {title}, context: {context}", return_tensors="pt"
        # paraphrase maybe?
        f"summarize: {context}", return_tensors="pt"
    ).input_ids
    outputs = model.generate(input_ids)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result
