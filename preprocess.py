import os
import re
import argparse

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

def parse_company_name(file_name):
    names = []
    # print(file_name, re.split('-|_| ', file_name.replace("_", " ")))
    for name in re.split('-|_| ', file_name.replace("_", " ")):
        if name.upper() in ["ESG", "SR", "AR", "CSR", "REPORT", "APPENDIX"]:
            continue
        if name.upper().endswith(("PDF", "TXT", "TSV")):
            continue
        try:
            val = int(name)
            if val > 2000:
                continue
        except ValueError as err:
            pass

        name = name[0].upper() + name[1:] if len(name) > 1 else name.upper()
        names.append(name)
    company_name = "_".join(names)
    return company_name

def reformat_es(input_path):
    """genereate elasticsearch index format file"""
    # file name format = <company name> + _ + <type,timestamp> + .txt
    file_name = input_path.replace(".txt", "").split("/")[-1]
    company_name = parse_company_name(file_name)
    output_path = input_path.replace(".txt", ".tsv")

    with open(input_path, "r", encoding="utf8") as fp:
        data = [to_third_person(line.strip(), company_name) for line in fp.readlines()]

    with open(output_path, "w", encoding="utf8") as fp:
        for line in data:
            fp.write(f"{company_name}\t{line}\n")


def to_third_person(sentence, entity):
    """change the sentence into third person"""
    result = sentence.split(" ")
    for i, word in enumerate(result):
        if word.lower() in ["i", "me", "myself", "we", "ourself", "ourselves"]:
            result[i] = entity
        elif word.lower() in ["my", "our", "ours", "mine"]:
            result[i] = f"{entity}'s"
    return " ".join(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/", type=str, required=True,
                        help="original raw files(in PDF format)")
    parser.add_argument("--output_dir", default="output/", type=str, required=True,
                        help="pre-processed files(in text format)")
    parser.add_argument("--to_es", default=True, type=bool, required=False,
                        help="generate es formatted files(in tsv format)")
    args = parser.parse_args()
    
    for file in tqdm.tqdm(os.listdir(args.data_dir)):
        input_path = os.path.join(args.data_dir, file)
        output_path = os.path.join(args.output_dir, file.split(".")[0]+".txt")

        extract_pdf_using_textract(input_path, output_path="temp1.txt")
        preprocess_txt_doc(input_path="temp1.txt", output_path="temp2.txt")
        extract_important_paragraph(input_path="temp2.txt", output_path=output_path)
        if args.to_es:
            reformat_es(input_path=output_path)


if __name__ == "__main__":
    main()
