import json
import zipfile
import xml.etree.ElementTree as ET
import sys
import re

def get_docx_text(path):
    try:
        with zipfile.ZipFile(path) as docx:
            tree = ET.parse(docx.open('word/document.xml'))
            root = tree.getroot()
            texts = []
            for node in root.iter():
                if node.tag.endswith('}t'):
                    if node.text:
                        texts.append(node.text)
            return '\n'.join(texts)
    except Exception as e:
        return f"Error reading {path}: {e}"

with open('extracted_info.txt', 'w', encoding='utf-8') as out:
    out.write("==== Ivy Masters Project specification form 25-26 ====\n")
    out.write(get_docx_text('Ivy Masters Project specification form 25-26 .docx'))
    out.write("\n\n==== B01800450 Research Design ====\n")
    text = get_docx_text('B01800450 Research Design.docx')
    # writing only first 5000 chars to avoid huge file
    if len(text) > 5000:
        out.write(text[:5000] + "\n...[TRUNCATED]...\n")
    else:
        out.write(text)
    
    out.write("\n\n==== Ivy_Project.ipynb Key Cells ====\n")
    try:
        with open('Ivy_Project.ipynb', 'r', encoding='utf-8') as f:
            nb = json.load(f)
        for cell in nb.get('cells', []):
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                if re.search(r'(?i)(features|columns|x_train|streamlit|input|categorical|numerical|drop)', source):
                    out.write("\n--- CODE CELL ---\n")
                    out.write(source)
    except Exception as e:
        out.write(f"Error reading notebook: {e}\n")
    
print("Extraction complete.")
