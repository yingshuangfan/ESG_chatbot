
from PyPDF2 import PdfReader
 
def extract_pdf(input_path, output_path):
    with open(output_path, "w", encoding="utf8") as out:
        try:
            reader = PdfReader(input_path)
            print(f"start parsing file: {input_path}, total {len(reader.pages)} pages.")
            for page in reader.pages:
                
                parts = []
                def visitor_body(text, cm, tm, font_dict, font_size):
                    y = tm[5]
                    if y > 50 and y < 720:  # igore header and footer
                        parts.append(text.strip() + " ")  # insert line-break
            

                page.extract_text(visitor_text=visitor_body)
                text_body = "".join(parts)
                out.write(text_body + "\n\n")   # insert page-break
        except Exception as err:
            print(f"failed to parse PDF {input_path} with error: {err}")

def test():
    ex_in = "../data/HSBC_HK_ESG_2022.pdf"
    ex_out = "../data/txt/HSBC_HK_ESG_2022.txt"
    extract_pdf(ex_in, ex_out)


if __name__ == "__main__":
    test()
