document.getElementById('uploadButton').addEventListener('click', () => {
    document.getElementById('imageInput').click();
});

document.getElementById('imageInput').addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload a valid image file.');
            return;
        }
        
        const reader = new FileReader();
        reader.onload = () => {
            document.getElementById('selectedImage').src = reader.result;
            document.getElementById('selectedImage').style.display = 'block';
            uploadImage(file);
        };
        reader.readAsDataURL(file);
    }
});

async function uploadImage(file) {
    document.getElementById('loading').style.display = 'block';

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        document.getElementById('artwork').textContent = `Name: ${data.artwork}`;

        document.getElementById('prediction').style.display = 'block';
    } catch (error) {
        console.error('Error uploading image:', error);
    } finally {
        document.getElementById('loading').style.display = 'none';
    }
}