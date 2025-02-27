const video = document.getElementById('video');  // The video element to display webcam feed
const resultText = document.getElementById('result-text');  // Element to display the result
const startRecognitionBtn = document.getElementById('start-recognition');  // Button to start recognition

// Start the webcam feed
async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;  // Set the video source to the webcam stream
    } catch (error) {
        console.error("⚠️ Camera access error:", error);
        alert("Could not access the camera. Please allow camera permissions.");
    }
}

// Capture an image from the video feed and convert it to a Blob
async function captureImage() {
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    canvas.width = 28;  // Resize to match the model input size (28x28)
    canvas.height = 28;
    context.drawImage(video, 0, 0, 28, 28);  // Capture the current frame from video feed

    return new Promise((resolve) => {
        canvas.toBlob((blob) => {
            resolve(blob);  // Resolve with the image blob
        }, 'image/png');
    });
}

// Send the captured image to the Flask backend for prediction
async function sendImageToBackend(imageBlob) {
    const formData = new FormData();
    formData.append('file', imageBlob, 'image.png');  // Append the image file to FormData

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        const result = await response.json();

        if (response.ok) {
            console.log('Prediction Result:', result.predicted_class);
            resultText.innerText = "Predicted Gesture: " + result.predicted_class;
        } else {
            console.error('Prediction failed:', result.error);
        }
    } catch (error) {
        console.error('Error sending image to backend:', error);
    }
}

// Start recognition when the button is clicked
startRecognitionBtn.addEventListener('click', async () => {
    await startCamera();  // Start the camera feed
    const imageBlob = await captureImage();  // Capture the image from the video feed
    await sendImageToBackend(imageBlob);  // Send the image to Flask for prediction
});

// Speech to text functionality
const speechResultText = document.getElementById('speech-text'); // Make sure this element exists in your HTML
const startSpeechRecognitionBtn = document.getElementById('start-speech'); // Button to trigger speech recognition

// Check if SpeechRecognition is supported by the browser
const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
recognition.lang = 'en-US'; // Set the language for speech recognition

// Event listener for capturing results
recognition.onresult = function(event) {
    const transcript = event.results[0][0].transcript;  // Get the transcript of the recognized speech
    console.log("Recognized Speech: ", transcript); // Log the result to the console
    speechResultText.innerText = transcript; // Display the recognized speech in the specified element
};

// Handle errors (e.g., no microphone, permission issues, etc.)
recognition.onerror = function(event) {
    console.error('Speech recognition error', event.error);
    alert("An error occurred with speech recognition: " + event.error);
};

// Optionally, handle end of speech recognition
recognition.onend = function() {
    console.log("Speech recognition has ended.");
};

// Start speech recognition when the button is clicked
startSpeechRecognitionBtn.addEventListener('click', () => {
    try {
        recognition.start(); // Start listening for speech
        console.log('Speech recognition started...');
    } catch (error) {
        console.error('Speech recognition could not start:', error);
        alert('Speech recognition failed to start. Please check your browser or microphone settings.');
    }
});

// New JavaScript for "Learn ASL" Button and Gallery
document.getElementById("learn-asl-button").addEventListener("click", function() {
  // Get the ASL gallery
  const gallery = document.getElementById("asl-gallery");
  
  // Toggle the hidden class to show or hide the gallery
  gallery.classList.toggle("hidden");
  
  // Optionally, change button text when the gallery is shown
  if (gallery.classList.contains("hidden")) {
    this.textContent = "Learn ASL";  // Button text when gallery is hidden
  } else {
    this.textContent = "Hide ASL Gallery";  // Button text when gallery is shown
  }
});
