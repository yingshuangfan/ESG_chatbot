"""
Generates prompt questions and answers with BERT
reference: https://huggingface.co/docs/transformers/v4.27.2/en/tasks/question_answering#inference
"""
import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer
from transformers import BertForQuestionAnswering, BertTokenizer
from transformers import pipeline

def load_sample_data(n_samples=100):
    """load few data from SQuAD"""
    samples = load_dataset("squad", split=f"train[:{n_samples}]")
    return samples

def fine_tune():
    """fine tune pre-trained BERT with prompt QA set"""
    model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased")

    args = TrainingArguments(
        output_dir="./model", 
        evaluation_strategy="epoch", 
        num_train_epochs=3
    )
    # TODO: prepare prompt set, define tokenizer and collator
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        data_collator=None,
    )
    trainer.train()

def inference(question, context):
    """generate answer given question and context"""
    # reference: https://huggingface.co/distilbert-base-cased-distilled-squad
    # tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
    # model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")
    checkpoint = "bert-large-uncased-whole-word-masking-finetuned-squad"
    tokenizer = BertTokenizer.from_pretrained(checkpoint)
    model = BertForQuestionAnswering.from_pretrained(checkpoint)

    agent = pipeline("question-answering", model=model, tokenizer=tokenizer)
    answer = agent(question=question, context=context)
    return answer

def test_squad():
    question = "How many programming languages does BLOOM support?"
    context = "BLOOM has 176 billion parameters and can generate text in 46 languages natural languages and 13 programming languages."
    print(inference(question, context))

def test_esg():
    # TODO: Split the report into several context, the maximum tokens (by default) is 512
    context = """
    Our approach to ESG We are on a journey to incorporate environmental, social and governance principles throughout the organisation, 
    and are taking  steps to embed sustainability into our purpose and corporate strategy.  
    About the ESG review  Our purpose is: ‘Opening up a world of opportunity’. 
    To achieve our purpose and deliver our strategy in a way that is sustainable, we are guided by our values: 
    we value difference; we succeed together; we take responsibility; and we get it done. 
    We also need to build strong relationships with all of our stakeholders, who are the people who work for us, bank with us, own us, regulate us, 
    and live in the societies we serve and on the planet we all inhabit.  
    Transition to net zero  We have continued to take steps to implement our climate ambition to become net zero in our operations and our supply chain by 2030, 
    and align our financed emissions to net zero by 2050. We have expanded our coverage of sectors for on-balance sheet financed emissions targets, 
    noting the challenge of evolving methodologies and data limitations. In addition, our operating environment for climate analysis and portfolio alignment is developing. 
    We continue work to improve our data management processes and are setting targets to align our provision of finance with the goals and timelines of the Paris Agreement. 
    In March 2022, we announced plans to turn our net zero ambition for our portfolio of clients into business transformation across the Group. 
    The plan involves the publication of a Group-wide climate transition plan in 2023.
    """

    with open("questions.json", "r", encoding="utf8") as fp:
        questions = json.load(fp)
        for _, question in questions.items():
            ans = inference(question, context)
            print(f"Q: {question}\nA:{ans}\n")


if __name__ == "__main__":
    test_esg()
