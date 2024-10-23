import pypdf
import pdfplumber

def extract_text_pdf(fn, engin='pypdf'):
    assert engin in ['pdfplumber', 'pypdf']
    print(fn)
    text = ''
    if engin == 'pdfplumber':
        with pdfplumber.open(fn) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + '\n'
    elif engin == 'pypdf':
        reader = pypdf.PdfReader(fn)
        for i in range(len(reader.pages)):
            page = reader.pages[i]
            text += page.extract_text() + '\n'
    return text


with open('10.5334_johd.1.txt', 'w', encoding='utf-8') as f:
    f.write(extract_text_pdf(r'data\data_papers\10.5334_johd.1.pdf'))