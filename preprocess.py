import os
import re
import tqdm
import textract
from PyPDF2 import PdfReader


keywords = set(["environment", "environmental", "sustainable", "sustainability", \
            "biodiversity", "renewable", "green", "climate", "material", "energy", \
            "wind", "solar", "forest", "water", "waste", "carbon", "emission", \
            "effluent", "pollutant", "hazardous", "disposal", "reputation", "ethic", \
            "ethics", "labor", "employment", "employee", "compensation", "pay", \
            "occupational", "health", "safety", "equity", "equal", "fairness", \
            "transparent", "transparency", "bias", "training", "education", "diverse", \
            "diversity", "discrimination", "nondiscrimination", "freedom", "minority", \
            "woman", "compliance", "regulation", "planning", "value", "economic", \
            "financial", "business", "strategy", "performance", "risk", "innovation", \
            "stakeholder", "management", "process", "opportunity", "responsible", "responsibility"])


def extract_pdf(input_path, output_path):
    with open(output_path, "w", encoding="utf8") as fp:
        try:
            reader = PdfReader(input_path)
            print(f"start parsing file: {input_path}, total {len(reader.pages)} pages.")
            for page_number in range(len(reader.pages)):
                page = reader.pages[page_number]
                page_content = page.extract_text()
                fp.write(page_content)
        except Exception as err:
            print(f"failed to parse PDF {input_path} with error: {err}")


def extract_pdf_using_textract(input_path, output_path):
    with open(output_path, "wb") as fp:
        try:
            text = textract.process(input_path, method='pdfminer')
            fp.write(text)
        except Exception as err:
            print(f"failed to parse PDF {input_path} with error: {err}")


def extract_important_paragraph(input_path, output_path):
    fp = open(input_path, "r")
    with open(output_path, "w") as output_txt:
        for line in fp.readlines():
            line = re.sub('[\u4e00-\u9fa5]','',line)
            line = line.strip()+"\n"
            temp = line.split(" ")
            set_c = set(temp) & keywords
            list_c = list(set_c)
            if len(list_c) != 0:
                output_txt.write(line)                


def preprocess_txt_doc(input_path, output_path):
    fp = open(input_path, "r")
    output_txt = open(output_path, "w")
    temp = ""
    for line in fp.readlines():
        if len(line) == 1:
            if len(temp) > 150:
                output_txt.write(temp+"\n")
            temp = ""
        temp += line.rstrip("\n")
    fp.close()
    output_txt.close()


def main(input_path, output_path):
    extract_pdf_using_textract(input_path, output_path="temp1.txt")
    preprocess_txt_doc(input_path="temp1.txt", output_path="temp2.txt")
    extract_important_paragraph(input_path="temp2.txt", output_path=output_path)


if __name__ == "__main__":
    dataset_dir = "data/"
    output_dir = "output/"
    for file in tqdm.tqdm(os.listdir(dataset_dir)):
        input_path = os.path.join(dataset_dir, file)
        output_path = os.path.join(output_dir, file.split(".")[0]+".txt")
        main(input_path, output_path)
