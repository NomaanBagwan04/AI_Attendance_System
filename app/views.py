from flask import Blueprint, render_template, request, redirect, url_for
import os
import pandas as pd
from datetime import datetime
from app.utils import process_image  # Assuming process_image is defined in utils.py

bp = Blueprint('views', __name__)

@bp.route('/')
def index():
    # Load attendance records
    records_path = 'data/attendance_records/attendance_records.csv'
    if os.path.exists(records_path):
        records = pd.read_csv(records_path)
    else:
        records = pd.DataFrame(columns=['student_id', 'date', 'time', 'status', 'subject', 'class'])
    return render_template('index.html', records=records.to_dict(orient='records'))

@bp.route('/upload', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        subject = request.form['subject']
        class_level = request.form['class']
        file = request.files['attendance_image']
        if file:
            filepath = os.path.join('data/student_images', file.filename)
            file.save(filepath)
            # Process the image and update attendance records
            process_image(filepath, subject, class_level)
            return redirect(url_for('views.index'))
    return render_template('upload.html')
