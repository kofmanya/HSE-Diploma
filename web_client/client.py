from flask import Flask, request, render_template, jsonify, url_for, send_from_directory
import requests
import os

app = Flask(__name__)
#UPLOAD_FOLDER = "web_client/static/uploads" # Folder to store uploaded images
UPLOAD_FOLDER = "/app/static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Backend API URL (FastAPI)
#API_URL = "http://127.0.0.1:8000/search/"
API_URL = "http://backend:8000/search/"

# Static file serving route
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None
    image_url = None

    if request.method == "POST":
        file = request.files["file"]
        if file:
            try:
                # Save the uploaded file locally
                file_path = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(file_path)
                #image_url = url_for("static", filename=f"uploads/{file.filename}")
                image_url = f"/uploads/{file.filename}"

                # Send the file to the FastAPI backend
                with open(file_path, "rb") as f:
                    response = requests.post(API_URL, files={"file": f})
                    response_data = response.json()

                if "error" in response_data:
                    error = response_data["error"]
                else:
                    result = response_data

            except Exception as e:
                error = f"Error: {str(e)}"

    return render_template("index.html", result=result, error=error, image_url=image_url)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)
