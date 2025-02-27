const video = document.getElementById('video');
const resultText = document.getElementById('result-text');
const startRecognitionBtn = document.getElementById('start-recognition');

let isRecognizing = false; // Flag to control recognition process

// Start the webcam feed
async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        console.log("✅ Camera started successfully");
    } catch (error) {
        console.error("⚠️ Camera access error:", error);
        alert("Could not access the camera. Please allow camera permissions.");
    }
}

// Capture an image from the video feed and convert it to a Blob
async function captureImage() {
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    canvas.width = 28;
    canvas.height = 28;
    context.drawImage(video, 0, 0, 28, 28);

    return new Promise((resolve) => {
        canvas.toBlob((blob) => {
            resolve(blob);
        }, 'image/png');
    });
}

// Send the captured image to the Flask backend for prediction
async function sendImageToBackend(imageBlob) {
    const formData = new FormData();
    formData.append('file', imageBlob, 'image.png');

    try {
        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            body: formData
        });
        const result = await response.json();

        if (response.ok) {
            console.log('🔹 Predicted:', result.predicted_class);
            resultText.innerText = "Predicted Gesture: " + result.predicted_class;
        } else {
            console.error('❌ Prediction failed:', result.error);
        }
    } catch (error) {
        console.error('⚠️ Error sending image to backend:', error);
    }
}

// **Continuous Recognition Loop**
async function startRecognitionLoop() {
    if (isRecognizing) return; // Prevent multiple loops from running
    isRecognizing = true;

    await startCamera(); // Start camera before recognition loop

    setInterval(async () => {
        if (!isRecognizing) return; // Stop loop if recognition is turned off
        const imageBlob = await captureImage();
        await sendImageToBackend(imageBlob);
    }, 200); // Captures a frame every 200ms (5 FPS)
}

// Start recognition when the button is clicked
startRecognitionBtn.addEventListener('click', () => {
    if (!isRecognizing) {
        console.log("🚀 Starting real-time recognition...");
        startRecognitionLoop();
        startRecognitionBtn.innerText = "Stop Recognition";
    } else {
        console.log("🛑 Stopping recognition...");
        isRecognizing = false;
        startRecognitionBtn.innerText = "Start Recognition";
    }
});
