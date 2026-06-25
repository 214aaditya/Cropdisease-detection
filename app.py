from flask import Flask, render_template, request, redirect, send_from_directory
from flask import session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
import json
import uuid
import os
import tensorflow as tf
from tensorflow.keras.models import load_model 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

app = Flask(__name__)
app.secret_key = "your-secret-key"  # required for session
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)
app.config['UPLOAD_FOLDER'] = 'uploadimages'

# User Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

# Create tables
with app.app_context():
    db.create_all()


# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(160, 160, 3)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(38, activation='softmax')  # 38 classes for different diseases
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

#model save
model.save("models/crop_disease_detection_model_pwp.keras")

#model load
model_path = "models/crop_disease_detection_model_pwp.keras"
model = tf.keras.models.load_model(model_path)


#model summary
model.summary()

try:
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!✅")
except Exception as e:
    print("Error loading model:", str(e))


# --------- LABELS ----------
label = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Background_without_leaves', 'Blueberry___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy',
    'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust', 'Corn___Northern_Leaf_Blight',
    'Corn___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# --------- OPTIONAL: DISEASE INFO ----------
# Load plant disease info JSON once
try:
    with open("plant_disease.json", 'r') as file:
        plant_disease_info = json.load(file)
except Exception as e:
    plant_disease_info = []
    print("Warning⚠️: Could not load plant_disease.json -", str(e))


# --------- ROUTES ----------

@app.route('/uploadimages/<path:filename>')
def uploaded_images(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


#----About section----

@app.route("/index.html",  methods=['GET'])
def index_page():  
    return render_template('index.html')

@app.route("/about.html",  methods=['GET'])
def about():
    return render_template('about.html')
#----features----
@app.route("/features.html")
def features():
    return render_template('features.html')
#----Contact----
@app.route("/contact.html")
def contact():
    return render_template("contact.html")
#----user-login----

@app.route("/userlogin.html", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        # Dummy authentication
        if email == "user@example.com" and password == "12345":
            flash("Login successful!", "success")
            return redirect("index.html")
        else:
            flash("Invalid credentials. Try again.", "danger")
            return redirect("userlogin.html")

    return render_template("userlogin.html")

#----registration----
@app.route("/register.html", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form["email"]
        password = generate_password_hash(request.form["password"])

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash("Email already registered.", "warning")
            return redirect("register.html")

        new_user = User(email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
        flash("Registration successful. Please log in.", "success")
        return redirect("userlogin.html")
    
    return render_template("register.html")
#----userlogin session----
@app.route("/userlogin.html", methods=["GET", "POST"])
def user_login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            session["user_id"] = user.id
            session["email"] = user.email
            flash("Logged in successfully.", "success")
            return redirect("index.html")
        else:
            flash("Invalid credentials.", "danger")
            return redirect("register.html")

    return render_template("login.html")

#----logout session----
@app.route("/templates/index.html")
def logout():
    session.clear()
    flash("Logged out successfully.", "info")
    return redirect("index.html")
#----------------------------------------------------------------------------------------------------------------

# --------- IMAGE PROCESSING ----------
def extract_features(image_path):
    image = tf.keras.utils.load_img(image_path, target_size=(160, 160))
    img_array = tf.keras.utils.img_to_array(image)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
def model_predict(image_path):
    img = extract_features(image_path)
    prediction = model.predict(img)
    predicted_index = np.argmax(prediction)
    predicted_label = label[predicted_index]

    # Fetch disease info from list of dicts
    disease_data = next((item for item in plant_disease_info if item["name"] == predicted_label), None)

    if disease_data:
        cause = disease_data.get("cause", "Cause info not available.")
        cure = disease_data.get("cure", "Cure info not available.")
    else:
        cause = "Cause info not available."
        cure = "Cure info not available."

    return predicted_label, cause, cure

#-------------------------------------
@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'img' not in request.files:
            flash("No image uploaded.", "warning")
            return redirect('/')

        image = request.files['img']
        if image.filename == '':
            flash("Empty filename.", "warning")
            return redirect('/')

        filename = f"temp_{uuid.uuid4().hex}_{image.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(filepath)

        predicted_label, cause, cure = model_predict(filepath)

        return render_template('index.html',
                               result=True,
                               imagepath=f'/uploadimages/{filename}',
                               prediction=predicted_label,
                               cause=cause,
                               cure=cure)
    else:
        return redirect('/')

# --------- MAIN ----------
if __name__ == "__main__":
    app.run(debug=True)
