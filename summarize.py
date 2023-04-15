"""
Generate summarized report abstract with T5
reference: https://huggingface.co/docs/transformers/v4.27.2/en/tasks/summarization#inference
"""

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def inference(text):
    # checkpoint = "t5-small"
    # reference: https://huggingface.co/philschmid/bart-large-cnn-samsum
    checkpoint = "philschmid/flan-t5-base-samsum"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    agent = pipeline("summarization", model=model, tokenizer=tokenizer)
    summary = agent(f"summarize: {text}")
    return summary[0]["summary_text"]

def inference_long(text):
    checkpoint = "pszemraj/led-base-book-summary"
    agent = pipeline("summarization", checkpoint)
    summary = agent(
           text,
           min_length=8, 
           max_length=512,
           no_repeat_ngram_size=3, 
           encoder_no_repeat_ngram_size=3,
           repetition_penalty=3.5,
           num_beams=4,
           do_sample=False,
           early_stopping=True,
    )
    return summary[0]["summary_text"]


def test_bill():
    text = """
    The Inflation Reduction Act lowers prescription drug costs, health care costs, and energy costs. 
    It's the most aggressive action on tackling the climate crisis in American history, 
    which will lift up American workers and create good-paying, union jobs across the country. 
    It'll lower the deficit and ask the ultra-wealthy and corporations to pay their fair share. 
    And no one making under $400,000 per year will pay a penny more in taxes.
    """
    print(inference(text))

def test_esg():
    with open("dataset/page_0.txt", "r", encoding="utf8") as fp:
        text = fp.readlines()[0]
        print(f"t5-base:\n{inference(text)}\n")
        print(f"longform-base:\n{inference_long(text)}\n")

if __name__ == "__main__":
    test_esg()
