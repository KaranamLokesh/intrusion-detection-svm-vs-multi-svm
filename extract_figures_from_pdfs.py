import os

import fitz  # PyMuPDF

PDFS = [
    "Classifying_Cover_Crop_Residue_from_RGB_Images_A_Simple_SVM_versus_a_SVM_Ensemble.pdf",
    "Intrusion_Detection_Mechanism_for_Large_Scale_Networks_using_CNN-LSTM.pdf",
]

os.makedirs("extracted_figures", exist_ok=True)

for pdf_path in PDFS:
    doc = fitz.open(pdf_path)
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    for page_num in range(len(doc)):
        page = doc[page_num]
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            if pix.n < 5:  # this is GRAY or RGB
                out_path = f"extracted_figures/{base}_page{page_num + 1}_img{img_index + 1}.png"
                pix.save(out_path)
            else:  # CMYK: convert to RGB first
                pix = fitz.Pixmap(fitz.csRGB, pix)
                out_path = f"extracted_figures/{base}_page{page_num + 1}_img{img_index + 1}.png"
                pix.save(out_path)
            print(f"Extracted {out_path}")
