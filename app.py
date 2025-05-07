import torch
import json
import os # Import os module
import base64
from io import BytesIO
from PIL import Image # Import Image from PIL
from flask import Flask, request, jsonify, render_template, send_from_directory # Import send_from_directory
from flask_cors import CORS # Import CORS
import time # Import time for unique filenames

# Import necessary components from your diffusion model file
from simple_diffusion_model import SimpleUNet, sample, IMAGE_SIZE, NUM_CHANNELS, TIMESTEPS, TIME_EMB_DIM, LABEL_EMB_DIM

# --- Configuration ---
LABEL_MAP_PATH = 'minecraft_label_map.json'
MODEL_SAVE_PATH = 'simple_diffusion_model.pth'
GENERATED_IMAGES_DIR = 'gallery/generated_images' # Define directory for generated images

app = Flask(__name__, static_folder='gallery/static', template_folder='gallery')
CORS(app) # Enable CORS for all routes

# Ensure the generated images directory exists
os.makedirs(GENERATED_IMAGES_DIR, exist_ok=True)

# Load label map and model once on startup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
label_map = {}
reverse_label_map = {}
num_classes = 0
model = None

try:
    with open(LABEL_MAP_PATH, 'r') as f:
        label_map = json.load(f)
    num_classes = len(label_map)
    reverse_label_map = {v: k for k, v in label_map.items()}

    # Initialize model and load state dict
    model = SimpleUNet(num_classes=num_classes, time_emb_dim=TIME_EMB_DIM, label_emb_dim=LABEL_EMB_DIM).to(device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.eval() # Set model to evaluation mode
    print("Model and label map loaded successfully.")

except FileNotFoundError as e:
    print(f"Error loading file: {e}. Make sure '{LABEL_MAP_PATH}' and '{MODEL_SAVE_PATH}' exist.")
    model = None # Ensure model is None if loading fails
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {LABEL_MAP_PATH}")
    model = None
except Exception as e:
    print(f"Error initializing or loading model: {e}")
    model = None

# Route to serve generated images
@app.route('/generated_images/<filename>')
def generated_image(filename):
    """Serve a generated image file."""
    return send_from_directory(GENERATED_IMAGES_DIR, filename)


@app.route('/')
def index():
    """Serve the main HTML page."""
    return render_template('index.html')

@app.route('/get_labels', methods=['GET'])
def get_labels():
    """Return the label map."""
    return jsonify(label_map)

@app.route('/list_generated_images', methods=['GET'])
def list_generated_images():
    """List the files in the generated_images directory."""
    try:
        # List files in the generated images directory
        image_files = [f for f in os.listdir(GENERATED_IMAGES_DIR) if os.path.isfile(os.path.join(GENERATED_IMAGES_DIR, f))]
        # Return a list of URLs relative to the server root
        image_urls = [f'/generated_images/{filename}' for filename in image_files]
        return jsonify({"images": image_urls})
    except Exception as e:
        print(f"Error listing generated images: {e}")
        return jsonify({"error": "Error listing images"}), 500

@app.route('/generate', methods=['POST'])
def generate_image():
    """Generate an image based on the provided label ID and save it."""
    if model is None:
        return jsonify({"error": "Model not loaded. Check server logs."}), 500

    data = request.json
    label_id = data.get('label_id')

    if label_id is None:
        return jsonify({"error": "Missing 'label_id' in request body"}), 400

    try:
        label_id = int(label_id)
    except ValueError:
        return jsonify({"error": "'label_id' must be an integer"}), 400

    if label_id < 0 or label_id >= num_classes:
        return jsonify({"error": f"Invalid label ID {label_id}. Must be between 0 and {num_classes - 1}."}), 400

    print(f"Received request to generate image for label ID: {label_id}")

    try:
        # Generate a single image for the given label
        sample_labels = torch.full((1,), label_id, dtype=torch.long, device=device)
        generated_image_tensor = sample(model, (1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE), TIMESTEPS, device, labels=sample_labels)

        # Convert tensor to PIL Image
        img_array = generated_image_tensor[0].permute(1, 2, 0).cpu().numpy()
        img = Image.fromarray(img_array, 'RGB')

        # Generate a unique filename and save the image
        timestamp = int(time.time())
        label_name = reverse_label_map.get(label_id, f'label_{label_id}')
        # Sanitize label name for filename
        safe_label_name = label_name.replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')
        filename = f"{safe_label_name}_{timestamp}.png"
        filepath = os.path.join(GENERATED_IMAGES_DIR, filename)
        img.save(filepath, format="PNG")
        print(f"Image saved to {filepath}")

        # Return the URL to the saved image
        image_url = f'/generated_images/{filename}'
        return jsonify({"image_url": image_url})

    except Exception as e:
        print(f"Error during image generation: {e}")
        return jsonify({"error": "Error generating image"}), 500

if __name__ == '__main__':
    # Ensure the gallery directory exists for static files
    os.makedirs('gallery', exist_ok=True)
    # Run the Flask app
    app.run(debug=True) # Set debug=False for production
