from flask import Flask, request, redirect, url_for, send_from_directory
from markupsafe import Markup
from werkzeug.utils import secure_filename
import os
import cv2
import pytesseract
import numpy as np
import pandas as pd
import re
from collections import defaultdict

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


class LayoutAwareOCR:
    # Layout-aware OCR that detects form structure automatically
    
    def __init__(self, image_path):
        self.original = cv2.imread(image_path)
        if self.original is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")
        
        self.preprocessed = None
        self.binary = None
        self.layout_elements = []
        self.form_fields = defaultdict(str)
        self.medications = []
        
    def preprocess(self):
        # Minimal preprocessing
        # Resize to standard width for consistency
        h, w = self.original.shape[:2]
        target_w = 2000
        if w != target_w:
            scale = target_w / w
            new_h = int(h * scale)
            img = cv2.resize(self.original, (target_w, new_h), interpolation=cv2.INTER_CUBIC)
        else:
            img = self.original.copy()
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Light denoising
        gray = cv2.fastNlMeansDenoising(gray, h=8)
        
        # Adaptive threshold (better for varying lighting)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 21, 10
        )
        
        self.preprocessed = img
        self.binary = binary
        return binary
    
    def detect_layout(self):
        # Use Tesseract's built-in layout analysis
        # Check if image needs inversion
        avg_val = np.mean(self.binary)
        if avg_val < 127:
            img_to_analyze = cv2.bitwise_not(self.binary)
        else:
            img_to_analyze = self.binary
        
        # PSM 3: Fully automatic page segmentation (best for layout detection)
        try:
            data = pytesseract.image_to_data(
                img_to_analyze,
                config='--oem 3 --psm 3',
                output_type=pytesseract.Output.DICT
            )
        except Exception as e:
            print(f"Layout detection failed: {e}")
            return []
        
        # Extract layout elements with bounding boxes
        n_boxes = len(data['text'])
        elements = []
        
        for i in range(n_boxes):
            text = data['text'][i].strip()
            if not text:
                continue
            
            conf = int(data['conf'][i])
            if conf < 0:  # Skip invalid confidence
                continue
            
            x = data['left'][i]
            y = data['top'][i]
            w = data['width'][i]
            h = data['height'][i]
            block_num = data['block_num'][i]
            par_num = data['par_num'][i]
            line_num = data['line_num'][i]
            
            elements.append({
                'text': text,
                'conf': conf,
                'bbox': (x, y, w, h),
                'block': block_num,
                'par': par_num,
                'line': line_num,
                'area': w * h
            })
        
        self.layout_elements = elements
        return elements
    
    def detect_form_structure(self):
        # Identify form fields using layout analysis
        if not self.layout_elements:
            self.detect_layout()
        
        # Group elements by lines (same block, par, line)
        lines = defaultdict(list)
        for elem in self.layout_elements:
            key = (elem['block'], elem['par'], elem['line'])
            lines[key].append(elem)
        
        # Sort elements in each line by x-coordinate
        for key in lines:
            lines[key].sort(key=lambda e: e['bbox'][0])
        
        # Detect field patterns
        for line_key, line_elems in lines.items():
            line_text = ' '.join([e['text'] for e in line_elems])
            
            # Patient name pattern - look for "FOR" field
            if re.search(r'FOR.*name.*address', line_text, re.I):
                # Next block likely contains patient info
                next_block = line_key[0] + 1
                patient_elems = [e for e in self.layout_elements 
                               if e['block'] == next_block]
                if patient_elems:
                    self.form_fields['patient_name'] = ' '.join(
                        [e['text'] for e in sorted(patient_elems, 
                         key=lambda x: x['bbox'][0])]
                    )
            
            # Also try to find patient name in top 20% of document
            if not self.form_fields['patient_name']:
                h = self.preprocessed.shape[0]
                top_elems = [e for e in self.layout_elements 
                           if e['bbox'][1] < 0.2 * h and e['conf'] > 50]
                if top_elems:
                    # Get largest text block in top area
                    largest = max(top_elems, key=lambda x: x['area'])
                    self.form_fields['patient_name'] = largest['text']
            
            # Date pattern
            if re.search(r'DATE', line_text, re.I):
                # Find text in same line or nearby with date pattern
                date_candidates = [e['text'] for e in line_elems 
                                 if re.search(r'\d{1,2}.*\d{2,4}', e['text'])]
                if date_candidates:
                    self.form_fields['date'] = date_candidates[0]
            
            # Also search for standalone date patterns
            if not self.form_fields['date']:
                if re.search(r'\d{1,2}[/-]\w+[/-]\d{2,4}', line_text):
                    self.form_fields['date'] = line_text
        
        return self.form_fields
    
    def detect_table_structure(self):
        # Detect table rows and columns using contour analysis
        # Detect horizontal lines
        h, w = self.binary.shape
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w//30, 1))
        horizontal_lines = cv2.morphologyEx(self.binary, cv2.MORPH_OPEN, 
                                           horizontal_kernel, iterations=2)
        
        # Detect vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h//30))
        vertical_lines = cv2.morphologyEx(self.binary, cv2.MORPH_OPEN, 
                                         vertical_kernel, iterations=2)
        
        # Find line intersections (cell corners)
        table_structure = cv2.bitwise_and(horizontal_lines, vertical_lines)
        
        # Find contours of cells
        contours, _ = cv2.findContours(table_structure, cv2.RETR_TREE, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        cells = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 50 and h > 20:  # Filter small noise
                cells.append({'bbox': (x, y, w, h), 'content': []})
        
        # Sort cells by position (top-to-bottom, left-to-right)
        cells.sort(key=lambda c: (c['bbox'][1], c['bbox'][0]))
        
        return cells, horizontal_lines, vertical_lines
    
    def extract_medications_from_table(self):
        # Extract medication information using detected table structure
        cells, h_lines, v_lines = self.detect_table_structure()
        
        if not cells:
            return []
        
        # Assign text elements to cells
        for elem in self.layout_elements:
            ex, ey, ew, eh = elem['bbox']
            elem_center_x = ex + ew/2
            elem_center_y = ey + eh/2
            
            for cell in cells:
                cx, cy, cw, ch = cell['bbox']
                if (cx <= elem_center_x <= cx+cw and 
                    cy <= elem_center_y <= cy+ch):
                    cell['content'].append(elem)
                    break
        
        # Group cells into rows
        rows = defaultdict(list)
        for cell in cells:
            if cell['content']:
                row_y = cell['bbox'][1]
                rows[row_y].append(cell)
        
        # Sort cells within each row by x-coordinate
        medications = []
        for row_y in sorted(rows.keys()):
            row_cells = sorted(rows[row_y], key=lambda c: c['bbox'][0])
            
            row_texts = []
            for cell in row_cells:
                cell_text = ' '.join([e['text'] for e in 
                                     sorted(cell['content'], 
                                           key=lambda x: x['bbox'][0])])
                row_texts.append(cell_text)
            
            # Expect at least 2 columns (medicine, dosage)
            if len(row_texts) >= 2:
                med_name = row_texts[0]
                dosage = row_texts[1] if len(row_texts) > 1 else ''
                frequency = row_texts[2] if len(row_texts) > 2 else ''
                
                # Filter out likely headers/labels
                if (med_name and 
                    not re.search(r'^(inscription|subscription|signa|rx|medicine|dosage|frequency)', 
                                 med_name, re.I) and
                    len(med_name) > 2):
                    medications.append({
                        'medicine': self.normalize_text(med_name),
                        'dosage': dosage,
                        'frequency': frequency
                    })
        
        self.medications = medications
        return medications
    
    def normalize_text(self, text):
        # Clean and normalize extracted text
        if not text:
            return ''
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep alphanumeric and common punctuation
        text = re.sub(r'[^\w\s\-\./()]', '', text)
        return text.strip()
    
    def fallback_medication_extraction(self):
        # Fallback: region-based extraction if table detection fails
        h, w = self.binary.shape
        
        # Define likely prescription area (middle section)
        rx_region = self.binary[int(0.25*h):int(0.75*h), :]
        
        # Check if image needs inversion
        avg_val = np.mean(rx_region)
        if avg_val < 127:
            rx_region = cv2.bitwise_not(rx_region)
        
        # Try multiple PSM modes for handwriting
        best_result = []
        for psm in [3, 4, 6, 11, 13]:  # PSM 13 is for raw line (good for handwriting)
            try:
                text = pytesseract.image_to_string(
                    rx_region,
                    config=f'--oem 3 --psm {psm}'
                )
                lines = [l.strip() for l in text.split('\n') if l.strip()]
                if len(lines) > len(best_result):
                    best_result = lines
            except:
                continue
        
        # Parse lines into medication entries
        medications = []
        for line in best_result:
            # Skip obvious non-medication lines
            if re.search(r'^(rx|inscription|subscription|signa|date|for)', line, re.I):
                continue
            
            # Look for patterns like "Medicine XYZ 10mg"
            parts = line.split()
            if len(parts) >= 1 and len(line) > 3:
                # Assume first part(s) are medicine name
                # Last part with digits might be dosage
                dosage_idx = -1
                for i, part in enumerate(parts):
                    if re.search(r'\d+\s*(mg|ml|g|mcg|gm)', part, re.I):
                        dosage_idx = i
                        break
                
                if dosage_idx > 0:
                    med_name = ' '.join(parts[:dosage_idx])
                    dosage = ' '.join(parts[dosage_idx:])
                    medications.append({
                        'medicine': self.normalize_text(med_name),
                        'dosage': dosage,
                        'frequency': ''
                    })
                else:
                    medications.append({
                        'medicine': self.normalize_text(line),
                        'dosage': '',
                        'frequency': ''
                    })
        
        if not self.medications:  # Only use fallback if main method failed
            self.medications = medications
        
        return medications
    
    def process(self):
        # Main processing pipeline (preprocess -> layout -> fields -> meds)
        self.preprocess()
        self.detect_layout()
        self.detect_form_structure()
        self.extract_medications_from_table()
        if not self.medications:
            self.fallback_medication_extraction()
        if not self.medications:
            self.last_resort_extraction()
        return self.create_results()
    
    def last_resort_extraction(self):
        # Last resort: extract all visible text
        avg_val = np.mean(self.binary)
        if avg_val < 127:
            img_to_ocr = cv2.bitwise_not(self.binary)
        else:
            img_to_ocr = self.binary
        
        try:
            text = pytesseract.image_to_string(img_to_ocr, config='--oem 3 --psm 3')
            lines = [l.strip() for l in text.split('\n') if l.strip() and len(l.strip()) > 2]
            
            for line in lines:
                if not re.search(r'^(rx|for|date|medical|signature)', line, re.I):
                    self.medications.append({
                        'medicine': self.normalize_text(line),
                        'dosage': '',
                        'frequency': ''
                    })
        except:
            pass
    
    def create_results(self):
        # Build DataFrame and return patient/date
        df = pd.DataFrame(self.medications) if self.medications else pd.DataFrame(columns=['Medicine', 'Dosage', 'Frequency'])
        if not self.medications:
            df.columns = ['Medicine', 'Dosage', 'Frequency']
        patient_name = self.form_fields.get('patient_name', 'Not detected')
        date_text = self.form_fields.get('date', 'Not detected')
        return patient_name, date_text, df


def process_image(image_path):
    """Process prescription image using layout-aware OCR"""
    try:
        ocr = LayoutAwareOCR(image_path)
        patient_name, date_text, df = ocr.process()
        
        # Save CSV
        csv_name = os.path.splitext(image_path)[0] + "_output.csv"
        df.to_csv(csv_name, index=False)
        
        # Save to DB
        create_table()
        for index, row in df.iterrows():
            try:
                save_row(patient_name, date_text, row["Medicine"], row["Dosage"], row["Frequency"])
            except Exception as e:
                print(f"DB save failed for row {index}: {e}")
        
        return patient_name, date_text, df, csv_name
    
    except Exception as e:
        print(f"Processing error: {e}")
        raise


@app.route('/')
def index():
    # Simple upload form with basic CSS
    html = '''
    <!doctype html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Layout-Aware Prescription OCR</title>
        <style>
            body { font-family: Arial, sans-serif; background:#f5f7fb; color:#222; }
            .card { max-width:760px; margin:40px auto; background:white; padding:20px; border-radius:8px; box-shadow:0 6px 18px rgba(20,30,60,0.08); }
            h1 { margin-top:0; color:#2b8aef; }
            .badge { display:inline-block; background:#e3f2fd; color:#1976d2; padding:4px 10px; border-radius:4px; font-size:0.85em; font-weight:600; }
            input[type=file] { display:block; margin:12px 0; }
            button { background:#2b8aef; color:white; border:none; padding:10px 14px; border-radius:6px; cursor:pointer; font-size:15px; }
            button:hover { background:#1976d2; }
            .features { margin-top:20px; padding:15px; background:#f8f9fa; border-radius:6px; }
            .features li { margin:8px 0; }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>📋 Prescription OCR <span class="badge">Layout-Aware</span></h1>
            <form method="post" action="/process" enctype="multipart/form-data">
                <label><strong>Upload prescription image (png/jpg)</strong></label>
                <input type="file" name="image" accept="image/*" required />
                <button type="submit">🔍 Upload & Process</button>
            </form>
            
            <div class="features">
                <strong>✨ Features:</strong>
                <ul>
                    <li>✓ Automatic layout detection (no hardcoded positions)</li>
                    <li>✓ Intelligent form field recognition</li>
                    <li>✓ Smart table structure detection</li>
                    <li>✓ Multi-level fallback for challenging documents</li>
                    <li>✓ Handwriting support</li>
                </ul>
            </div>
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

        # Build simple HTML response showing results
        table_html = df.to_html(index=False, classes='result-table')
        safe_img_url = url_for('uploaded_file', filename=filename)
        csv_basename = os.path.basename(csv_name)
        csv_url = url_for('uploaded_file', filename=csv_basename)
        
        # Add status indicators
        status_color = '#4caf50' if len(df) > 0 else '#ff9800'
        status_text = f'{len(df)} medications extracted' if len(df) > 0 else 'No medications found'
        
        resp = f"""
        <!doctype html>
        <html>
        <head>
            <meta charset='utf-8'>
            <title>OCR Results - Layout Aware</title>
            <style>
                body{{ font-family: Arial, sans-serif; background:#f5f7fb; color:#222; }}
                .card{{ max-width:1000px; margin:24px auto; background:white; padding:24px; border-radius:8px; box-shadow:0 6px 18px rgba(20,30,60,0.06); }}
                img.preview{{ max-width:380px; border:1px solid #eee; border-radius:6px; }}
                .meta{{ margin-bottom:16px; padding:12px; background:#f8f9fa; border-radius:6px; }}
                .meta-item{{ margin:6px 0; }}
                .status{{ display:inline-block; padding:4px 10px; border-radius:4px; font-size:0.9em; font-weight:600; background:{status_color}; color:white; }}
                table.result-table{{ border-collapse:collapse; width:100%; margin-top:12px; }}
                table.result-table th{{ background:#2b8aef; color:white; padding:10px; text-align:left; }}
                table.result-table td{{ border:1px solid #ddd; padding:8px; }}
                table.result-table tr:nth-child(even){{ background:#f8f9fa; }}
                .actions{{ margin-top:16px; }}
                .btn{{ display:inline-block; padding:8px 14px; margin-right:10px; background:#2b8aef; color:white; text-decoration:none; border-radius:6px; }}
                .btn:hover{{ background:#1976d2; }}
                .btn-secondary{{ background:#6c757d; }}
                .btn-secondary:hover{{ background:#5a6268; }}
            </style>
        </head>
        <body>
            <div class='card'>
                <h2>📋 OCR Results <span class="status">{status_text}</span></h2>
                
                <div class='meta'>
                    <div class='meta-item'><strong>👤 Patient:</strong> {Markup(patient_name)}</div>
                    <div class='meta-item'><strong>📅 Date:</strong> {Markup(date_text)}</div>
                    <div class='meta-item'><strong>🔍 Method:</strong> Layout-Aware Detection</div>
                </div>
                
                <div style='display:flex; gap:24px; flex-wrap:wrap;'>
                    <div>
                        <h3>Original Image</h3>
                        <img src='{safe_img_url}' class='preview' alt='uploaded' />
                    </div>
                    <div style='flex:1; min-width:400px;'>
                        <h3>Extracted Medications</h3>
                        {table_html}
                        <div class='actions'>
                            <a href='{csv_url}' class='btn'>📥 Download CSV</a>
                            <a href='/' class='btn btn-secondary'>🔄 Process Another</a>
                        </div>
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