function displayFileInfo() {
        const fileInput = document.getElementById('fileInput');
        const fileInfo = document.getElementById('fileInfo');
        const uploadIcon = document.getElementById('uploadIcon');
        const dropZone = document.getElementById('dropzone_upload_bulk');

        if (fileInput.files.length > 0) {
            const file = fileInput.files[0];
            fileInfo.textContent = `${file.name} (${(file.size / 1024).toFixed(2)} KB)`;
            fileInfo.style.display = 'block';
            uploadIcon.style.display = 'none';
            dropZone.style.border = 'none'
        }
    }
function uploadFile() {
    const fileInput = document.getElementById('fileInput');
    if (fileInput.files.length === 0) {
        alert('Please select a file to upload.');
        return;
    }

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    fetch('/bulk_predict', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (response.ok) {
            return response.blob();  // Convert the response to a blob
        } else {
            throw new Error('Error uploading file.');
        }
    })
    .then(blob => {
        // Create a blob URL and use it to trigger a download
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'results.csv';  // Suggested filename for the download
        document.body.appendChild(a);
        a.click();
        a.remove();
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Error uploading file.');
    });
}
