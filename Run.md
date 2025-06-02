## How to run the Flask application

1. **Open Command Prompt in your project directory**:
```bash
cd "C:\Users\Redab\OneDrive\Desktop\Dementia Classifier"
```

2. **Activate your virtual environment**:
```bash
venv\Scripts\activate
```

3. **Install Flask if not already installed**:
```bash
pip install Flask
```

4. **Run the Flask application**:
```bash
python flask_app.py
```

5. **Open your browser and go to**:
```
http://localhost:5000
```

## Summary of files you need to create:

1. **`flask_app.py`** - Main Flask web application (create in root directory)
2. **`templates/base.html`** - Base HTML template that other templates extend
3. **`static/uploads/`** - Directory for uploaded images
4. **`static/css/`** - Directory for custom CSS files (optional)
5. **`static/js/`** - Directory for custom JavaScript files (optional)

Once you create these files and run `python flask_app.py`, you'll have a fully functional web application that:

- ✅ Allows patient and doctor registration/login
- ✅ Lets patients upload MRI scans for AI analysis
- ✅ Provides doctor interface to review and confirm AI results
- ✅ Sends notifications between doctors and patients
- ✅ Uses SQLite database to store all data
- ✅ Integrates with your existing trained model
- ✅ Supports PNG, JPG, JPEG, ICO file formats

The demo doctor login is: **r@gov.nl** / **doctor123**