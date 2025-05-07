document.addEventListener('DOMContentLoaded', () => {
    const labelSelect = document.getElementById('label-select');
    const generateBtn = document.getElementById('generate-btn');
    const generatedImage = document.getElementById('generated-image');
    const placeholderText = document.getElementById('placeholder-text');
    const loadingDiv = document.getElementById('loading');
    const errorDiv = document.getElementById('error-message');
    const imageGalleryDiv = document.getElementById('image-gallery'); // Get the gallery div

    // Function to fetch labels and populate the dropdown
    async function fetchLabels() {
        try {
            const response = await fetch('/get_labels');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const labelMap = await response.json();
            // Clear existing options
            labelSelect.innerHTML = '';
            // Populate dropdown with labels
            for (const label in labelMap) {
                const option = document.createElement('option');
                option.value = labelMap[label]; // Use label ID as value
                option.textContent = label; // Use label name as text
                labelSelect.appendChild(option);
            }
        } catch (error) {
            console.error('Error fetching labels:', error);
            errorDiv.textContent = 'Error loading labels. Please check the server.';
            errorDiv.classList.remove('hidden');
        }
    }

    // Function to fetch and display generated images in the gallery
    async function fetchAndDisplayGallery() {
        try {
            const response = await fetch('/list_generated_images');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            if (data.images && Array.isArray(data.images)) {
                // Clear existing gallery images
                imageGalleryDiv.innerHTML = '';
                // Add each image to the gallery
                data.images.forEach(imageUrl => {
                    const imgElement = document.createElement('img');
                    imgElement.src = imageUrl;
                    imgElement.alt = 'Generated Texture';
                    imgElement.classList.add('gallery-image'); // Add a class for styling
                    imageGalleryDiv.appendChild(imgElement);
                });
            } else {
                console.warn('No images found in gallery.');
                imageGalleryDiv.innerHTML = '<p>No generated images yet.</p>';
            }
        } catch (error) {
            console.error('Error fetching gallery images:', error);
            // Optionally display an error in the gallery section
            imageGalleryDiv.innerHTML = `<p style="color: red;">Error loading gallery: ${error.message}</p>`;
        }
    }


    // Function to generate image
    async function generateImage() {
        const selectedLabelId = labelSelect.value;
        if (!selectedLabelId) {
            alert('Please select a texture type.');
            return;
        }

        // Show loading, hide image and placeholder
        loadingDiv.classList.remove('hidden');
        generatedImage.classList.add('hidden');
        placeholderText.classList.add('hidden');
        errorDiv.classList.add('hidden');

        try {
            const response = await fetch('/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ label_id: parseInt(selectedLabelId) }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            if (data.image_url) {
                // Add the newly generated image to the gallery
                const imgElement = document.createElement('img');
                imgElement.src = data.image_url;
                imgElement.alt = 'Generated Texture';
                imgElement.classList.add('gallery-image'); // Add a class for styling
                imageGalleryDiv.appendChild(imgElement); // Append to the gallery

                // Optionally, still show the latest generated image in the main image container
                generatedImage.src = data.image_url;
                generatedImage.style.display = 'block'; // Force display
                generatedImage.classList.remove('hidden'); // Also remove the class

            } else {
                throw new Error('No image URL received from server.');
            }

        } catch (error) {
            console.error('Error generating image:', error);
            errorDiv.textContent = `Error generating image: ${error.message}`;
            errorDiv.classList.remove('hidden');
            placeholderText.classList.remove('hidden'); // Show placeholder on error
        } finally {
            loadingDiv.classList.add('hidden'); // Hide loading regardless of success or failure
        }
    }

    // Event listener for the generate button
    generateBtn.addEventListener('click', generateImage);

    // Fetch labels and display gallery when the page loads
    fetchLabels();
    fetchAndDisplayGallery(); // Fetch and display existing images
});
