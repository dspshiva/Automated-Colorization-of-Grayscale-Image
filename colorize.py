from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploaded_image/'
OUTPUT_FOLDER = 'static/output_image/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Paths to Model Files
DIR = r"model_files/"
PROTOTXT = r"E:\Mini_Project(colorization)(1)\Mini_Project(colorization)\model_files\colorization_deploy_v2.prototxt"
MODEL = r"E:\Mini_Project(colorization)(1)\Mini_Project(colorization)\model_files\colorization_release_v2.caffemodel"
POINTS = os.path.join(DIR, "E:\Mini_Project(colorization)(1)\Mini_Project(colorization)\model_files\pts_in_hull.npy")

# Load Model
print("Loading model...")
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
pts = np.load(POINTS)
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            # Save the original file to the upload folder first
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Resize the image before colorizing
            image = cv2.imread(filepath)
            new_width = 500  # Desired width
            new_height = int(image.shape[0] * (new_width / image.shape[1]))  # Maintaining aspect ratio
            resized_image = cv2.resize(image, (new_width, new_height))

            # Save the resized image temporarily
            resized_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'resized_' + file.filename)
            cv2.imwrite(resized_filepath, resized_image)

            # Now colorize the resized image
            output_path = colorize_image(resized_filepath)
            return render_template('result.html', original=resized_filepath, output=output_path)
    return render_template('upload.html')


def colorize_image(image_path):
    # Load the input image
    image = cv2.imread(image_path)

    # Resize the input image to a fixed width while maintaining the aspect ratio
    new_width = 500  # Specify the desired width
    new_height = int(image.shape[0] * (new_width / image.shape[1]))
    resized_image = cv2.resize(image, (new_width, new_height))

    # Convert to LAB color space
    scaled = resized_image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    resized = cv2.resize(lab, (224, 224))  # Resize to the model's input size (224x224)
    L = cv2.split(resized)[0]
    L -= 50  # Adjust L channel for the model

    # Colorize the image
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    # Resize the 'ab' channel to match the input image's size
    ab = cv2.resize(ab, (resized_image.shape[1], resized_image.shape[0]))

    # Combine L and ab channels
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

    # Convert back to BGR color space
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")

    # Save the resized and colorized image
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], os.path.basename(image_path))
    cv2.imwrite(output_path, colorized)

    print(f"Colorized image saved to {output_path}")
    return output_path


if __name__ == '__main__':
    app.run(debug=True)