from flask import Flask, request, redirect, url_for, send_from_directory
from markupsafe import Markup
from werkzeug.utils import secure_filename
import os
import cv2
import pytesseract
import numpy as np
import pandas as pd
import re

from db import get_connection, create_table
from save_results import save_row

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Ensure pytesseract points to system tesseract (container/VM)
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'


def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_image(image_path):
        img = cv2.imread(image_path)
        if img is None:
                raise FileNotFoundError(image_path)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                             cv2.THRESH_BINARY_INV, 11, 2)

        h, w = th.shape
        patient_roi = th[40:110, 40:440]
        date_roi = th[40:110, 460:700]
        table_roi = th[140:600, 40:700]

        def ocr_image(binary_img, psm=6):
                inv = cv2.bitwise_not(binary_img)
                config = f'--oem 3 --psm {psm}'
                raw = pytesseract.image_to_string(inv, config=config)
                return re.sub(r'[\n\r\t\f\v]+', ' ', raw).strip()

        patient_name = ocr_image(patient_roi, psm=7)
        date_text = ocr_image(date_roi, psm=7)

        table_inv = cv2.bitwise_not(table_roi)
        contours, _ = cv2.findContours(table_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        row_entries = []
        for cnt in contours:
                x, y, wc, hc = cv2.boundingRect(cnt)
                if hc < 20 or wc < 50:
                        continue

                row_img = table_roi[y:y+hc, x:x+wc]
                col_w = max(1, wc // 3)
                med_img = row_img[:, 0:col_w]
                dose_img = row_img[:, col_w:2*col_w]
                freq_img = row_img[:, 2*col_w:wc]

                med_text = ocr_image(med_img, psm=7)
                dose_text = ocr_image(dose_img, psm=7)
                freq_text = ocr_image(freq_img, psm=7)

                row_entries.append((y, [med_text, dose_text, freq_text]))

        row_entries.sort(key=lambda tup: tup[0])
        rows_sorted = [tup[1] for tup in row_entries]

        if rows_sorted:
                df = pd.DataFrame(rows_sorted, columns=["Medicine", "Dosage", "Frequency"])
        else:
                df = pd.DataFrame(columns=["Medicine", "Dosage", "Frequency"]) 

        # save CSV
        csv_name = os.path.splitext(image_path)[0] + "_output.csv"
        df.to_csv(csv_name, index=False)

        # save to DB
        create_table()
        for index, row in df.iterrows():
                save_row(patient_name, date_text, row["Medicine"], row["Dosage"], row["Frequency"])

        return patient_name, date_text, df, csv_name


@app.route('/')
def index():
        # Simple upload form with basic CSS
        html = '''
        <!doctype html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Prescription OCR</title>
            <style>
                body { font-family: Arial, sans-serif; background:#f5f7fb; color:#222; }
                .card { max-width:760px; margin:40px auto; background:white; padding:20px; border-radius:8px; box-shadow:0 6px 18px rgba(20,30,60,0.08); }
                h1 { margin-top:0; }
                input[type=file] { display:block; margin:12px 0; }
                button { background:#2b8aef; color:white; border:none; padding:10px 14px; border-radius:6px; cursor:pointer; }
                .preview { max-width:320px; margin-top:12px; }
            </style>
        </head>
        <body>
            <div class="card">
                <h1>Prescription OCR</h1>
                <form method="post" action="/process" enctype="multipart/form-data">
                    <label>Upload prescription image (png/jpg)</label>
                    <input type="file" name="image" accept="image/*" required />
                    <button type="submit">Upload & Process</button>
                </form>
                <p style="font-size:0.9em;color:#555;margin-top:12px">After processing you will see extracted patient name, date and table.</p>
            </div>
        </body>
        </html>
        '''
        return html


@app.route('/process', methods=['POST'])
def process():
        if 'image' not in request.files:
                return redirect(url_for('index'))
        file = request.files['image']
        if file.filename == '':
                return redirect(url_for('index'))
        if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(save_path)
                try:
                        patient_name, date_text, df, csv_name = process_image(save_path)
                except Exception as e:
                        return f"Processing failed: {e}", 500

                # build simple HTML response showing results
                table_html = df.to_html(index=False, classes='result-table')
                safe_img_url = url_for('uploaded_file', filename=filename)
                csv_basename = os.path.basename(csv_name)
                csv_url = url_for('uploaded_file', filename=csv_basename)
                resp = f"""
                <!doctype html>
                <html>
                <head>
                    <meta charset='utf-8'>
                    <title>OCR Results</title>
                    <style>
                        body{{ font-family: Arial, sans-serif; background:#f5f7fb; color:#222; }}
                        .card{{ max-width:900px; margin:24px auto; background:white; padding:18px; border-radius:8px; box-shadow:0 6px 18px rgba(20,30,60,0.06); }}
                        img.preview{{ max-width:320px; border:1px solid #eee; border-radius:6px; }}
                        .meta{{ margin-bottom:12px; }}
                        table.result-table{{ border-collapse:collapse; width:100%; }}
                        table.result-table th, table.result-table td{{ border:1px solid #ddd; padding:8px; text-align:left; }}
                    </style>
                </head>
                <body>
                    <div class='card'>
                        <h2>OCR Results</h2>
                        <div class='meta'><strong>Patient:</strong> {Markup(patient_name)} &nbsp;&nbsp; <strong>Date:</strong> {Markup(date_text)}</div>
                        <div style='display:flex; gap:20px'>
                            <div>
                                <img src='{safe_img_url}' class='preview' alt='uploaded' />
                            </div>
                            <div style='flex:1'>
                                {table_html}
                                <p><a href='{csv_url}'>Download CSV</a></p>
                                <p><a href='/'>Process another image</a></p>
                            </div>
                        </div>
                    </div>
                </body>
                </html>
                """
                return resp
        return redirect(url_for('index'))


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
        # Run on 0.0.0.0 so it's reachable from host/devcontainer port forwarding
        app.run(host='0.0.0.0', port=5000, debug=True)

