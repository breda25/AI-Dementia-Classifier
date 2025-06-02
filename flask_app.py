from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import os
import uuid
from datetime import datetime
from src.inference import DementiaClassifier
import logging

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize the AI model - try multiple model paths for better accuracy
MODEL_PATHS = [
    'models/efficientnet_dementia_model.h5',   # High-accuracy EfficientNet model
    'models/resnet_dementia_model.h5',         # High-accuracy ResNet model  
    'models/checkpoint_01_0.7782.h5',          # Your current model (fallback)
]

classifier = None
for model_path in MODEL_PATHS:
    if os.path.exists(model_path):
        try:
            classifier = DementiaClassifier(model_path)
            print(f"✅ Loaded model: {model_path}")
            break
        except Exception as e:
            print(f"⚠️ Failed to load {model_path}: {e}")
            continue

if classifier is None:
    print("❌ No valid model found! Creating a basic model...")
    # Create a basic model as fallback
    from src.advanced_models import AdvancedDementiaModels
    basic_model = AdvancedDementiaModels.create_efficientnet_model()
    fallback_path = 'models/fallback_model.h5'
    basic_model.save(fallback_path)
    classifier = DementiaClassifier(fallback_path)
    print(f"✅ Created and loaded fallback model: {fallback_path}")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'ico'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init_db():
    """Initialize the SQLite database"""
    conn = sqlite3.connect('dementia_app.db')
    cursor = conn.cursor()
    
    # Users table (doctors and patients)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT NOT NULL,
            user_type TEXT NOT NULL CHECK(user_type IN ('doctor', 'patient')),
            phone TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Analysis results table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            image_filename TEXT NOT NULL,
            prediction_class TEXT NOT NULL,
            confidence_score REAL NOT NULL,
            probabilities TEXT NOT NULL,
            status TEXT DEFAULT 'pending' CHECK(status IN ('pending', 'confirmed', 'rejected')),
            doctor_id INTEGER,
            doctor_notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            reviewed_at TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES users (id),
            FOREIGN KEY (doctor_id) REFERENCES users (id)
        )
    ''')
    
    # Notifications table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS notifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            message TEXT NOT NULL,
            notification_type TEXT NOT NULL,
            is_read BOOLEAN DEFAULT FALSE,
            analysis_id INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id),
            FOREIGN KEY (analysis_id) REFERENCES analysis_results (id)
        )
    ''')
    
    # Create default doctor account
    cursor.execute('''
        INSERT OR IGNORE INTO users (email, password_hash, full_name, user_type, phone)
        VALUES (?, ?, ?, ?, ?)
    ''', ('doctor@hospital.com', generate_password_hash('doctor123'), 'Dr. Medical Expert', 'doctor', '+1234567890'))
    
    conn.commit()
    conn.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        full_name = request.form['full_name']
        user_type = request.form['user_type']
        phone = request.form.get('phone', '')
        
        conn = sqlite3.connect('dementia_app.db')
        cursor = conn.cursor()
        
        # Check if user already exists
        cursor.execute('SELECT id FROM users WHERE email = ?', (email,))
        if cursor.fetchone():
            flash('Email already registered')
            return render_template('register.html')
        
        # Create new user
        password_hash = generate_password_hash(password)
        cursor.execute('''
            INSERT INTO users (email, password_hash, full_name, user_type, phone)
            VALUES (?, ?, ?, ?, ?)
        ''', (email, password_hash, full_name, user_type, phone))
        
        conn.commit()
        conn.close()
        
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        conn = sqlite3.connect('dementia_app.db')
        cursor = conn.cursor()
        
        cursor.execute('SELECT id, password_hash, full_name, user_type FROM users WHERE email = ?', (email,))
        user = cursor.fetchone()
        conn.close()
        
        if user and check_password_hash(user[1], password):
            session['user_id'] = user[0]
            session['user_type'] = user[3]
            session['full_name'] = user[2]
            
            if user[3] == 'doctor':
                return redirect(url_for('doctor_dashboard'))
            else:
                return redirect(url_for('patient_dashboard'))
        else:
            flash('Invalid email or password')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/patient/dashboard')
def patient_dashboard():
    if 'user_id' not in session or session['user_type'] != 'patient':
        return redirect(url_for('login'))
    
    conn = sqlite3.connect('dementia_app.db')
    cursor = conn.cursor()
    
    # Get patient's analysis history
    cursor.execute('''
        SELECT ar.*, u.full_name as doctor_name
        FROM analysis_results ar
        LEFT JOIN users u ON ar.doctor_id = u.id
        WHERE ar.patient_id = ?
        ORDER BY ar.created_at DESC
    ''', (session['user_id'],))
    
    analyses = cursor.fetchall()
    
    # Get unread notifications
    cursor.execute('''
        SELECT * FROM notifications 
        WHERE user_id = ? AND is_read = FALSE 
        ORDER BY created_at DESC
    ''', (session['user_id'],))
    
    notifications = cursor.fetchall()
    conn.close()
    
    return render_template('patient_dashboard.html', analyses=analyses, notifications=notifications)

@app.route('/doctor/dashboard')
def doctor_dashboard():
    if 'user_id' not in session or session['user_type'] != 'doctor':
        return redirect(url_for('login'))
    
    conn = sqlite3.connect('dementia_app.db')
    cursor = conn.cursor()
    
    # Get pending analyses
    cursor.execute('''
        SELECT ar.*, u.full_name as patient_name, u.email as patient_email
        FROM analysis_results ar
        JOIN users u ON ar.patient_id = u.id
        WHERE ar.status = 'pending'
        ORDER BY ar.created_at ASC
    ''', )
    
    pending_analyses = cursor.fetchall()
    
    # Get recent analyses
    cursor.execute('''
        SELECT ar.*, u.full_name as patient_name
        FROM analysis_results ar
        JOIN users u ON ar.patient_id = u.id
        WHERE ar.doctor_id = ?
        ORDER BY ar.reviewed_at DESC
        LIMIT 10
    ''', (session['user_id'],))
    
    recent_analyses = cursor.fetchall()
    conn.close()
    
    return render_template('doctor_dashboard.html', 
                         pending_analyses=pending_analyses, 
                         recent_analyses=recent_analyses)

@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
    if 'user_id' not in session or session['user_type'] != 'patient':
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # Generate unique filename
            filename = str(uuid.uuid4()) + '.' + file.filename.rsplit('.', 1)[1].lower()
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                # Run AI analysis
                result = classifier.predict(filepath, display_result=False)
                
                if result['success']:
                    # Save to database
                    conn = sqlite3.connect('dementia_app.db')
                    cursor = conn.cursor()
                    
                    cursor.execute('''
                        INSERT INTO analysis_results 
                        (patient_id, image_filename, prediction_class, confidence_score, probabilities)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (session['user_id'], filename, result['class_name'], 
                          result['confidence'], str(result['probabilities'])))
                    
                    analysis_id = cursor.lastrowid
                    
                    # Notify doctors about new analysis
                    cursor.execute('SELECT id FROM users WHERE user_type = "doctor"')
                    doctors = cursor.fetchall()
                    
                    for doctor in doctors:
                        cursor.execute('''
                            INSERT INTO notifications (user_id, message, notification_type, analysis_id)
                            VALUES (?, ?, ?, ?)
                        ''', (doctor[0], f'New analysis from {session["full_name"]} requires review', 
                              'new_analysis', analysis_id))
                    
                    conn.commit()
                    conn.close()
                    
                    flash('Analysis completed successfully! Waiting for doctor review.')
                    return redirect(url_for('patient_dashboard'))
                else:
                    flash('Error analyzing image: ' + result.get('error', 'Unknown error'))
            
            except Exception as e:
                flash(f'Error processing image: {str(e)}')
        else:
            flash('Invalid file type. Please upload PNG, JPG, JPEG, or ICO files.')
    
    return render_template('MRIscan.html')

@app.route('/doctor/review/<int:analysis_id>', methods=['GET', 'POST'])
def review_analysis(analysis_id):
    if 'user_id' not in session or session['user_type'] != 'doctor':
        return redirect(url_for('login'))
    
    conn = sqlite3.connect('dementia_app.db')
    cursor = conn.cursor()
    
    if request.method == 'POST':
        status = request.form['status']
        notes = request.form['notes']
        
        # Update analysis
        cursor.execute('''
            UPDATE analysis_results 
            SET status = ?, doctor_id = ?, doctor_notes = ?, reviewed_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (status, session['user_id'], notes, analysis_id))
        
        # Get patient info for notification
        cursor.execute('''
            SELECT ar.patient_id, ar.prediction_class, u.full_name
            FROM analysis_results ar
            JOIN users u ON ar.patient_id = u.id
            WHERE ar.id = ?
        ''', (analysis_id,))
        
        patient_info = cursor.fetchone()
        
        if patient_info:
            # Create notification for patient
            if status == 'confirmed':
                message = f'Your analysis has been confirmed by Dr. {session["full_name"]}. Diagnosis: {patient_info[1]}'
            else:
                message = f'Your analysis has been reviewed by Dr. {session["full_name"]}. Please consult for more information.'
            
            cursor.execute('''
                INSERT INTO notifications (user_id, message, notification_type, analysis_id)
                VALUES (?, ?, ?, ?)
            ''', (patient_info[0], message, 'analysis_result', analysis_id))
        
        conn.commit()
        conn.close()
        
        flash('Analysis review completed successfully!')
        return redirect(url_for('doctor_dashboard'))
    
    # Get analysis details
    cursor.execute('''
        SELECT ar.*, u.full_name as patient_name, u.email as patient_email
        FROM analysis_results ar
        JOIN users u ON ar.patient_id = u.id
        WHERE ar.id = ?
    ''', (analysis_id,))
    
    analysis = cursor.fetchone()
    conn.close()
    
    if not analysis:
        flash('Analysis not found')
        return redirect(url_for('doctor_dashboard'))
    
    return render_template('analysis_review.html', analysis=analysis)

@app.route('/notifications/mark_read/<int:notification_id>')
def mark_notification_read(notification_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    conn = sqlite3.connect('dementia_app.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        UPDATE notifications SET is_read = TRUE 
        WHERE id = ? AND user_id = ?
    ''', (notification_id, session['user_id']))
    
    conn.commit()
    conn.close()
    
    return redirect(request.referrer or url_for('patient_dashboard'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)