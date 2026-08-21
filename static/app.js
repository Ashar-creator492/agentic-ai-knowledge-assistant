const uploadForm = document.getElementById('uploadForm');
const pdfInput = document.getElementById('pdfInput');
const fileLabel = document.getElementById('fileLabel');
const uploadButton = document.getElementById('uploadButton');
const uploadStatus = document.getElementById('uploadStatus');

const chatForm = document.getElementById('chatForm');
const questionInput = document.getElementById('questionInput');
const sendButton = document.getElementById('sendButton');
const chatMessages = document.getElementById('chatMessages');

pdfInput.addEventListener('change', () => {
  const file = pdfInput.files[0];
  fileLabel.textContent = file ? file.name : 'Choose a PDF to upload';
});

function addMessage(role, text, citation = '') {
  const row = document.createElement('div');
  row.className = `message-row ${role}`;

  const bubble = document.createElement('div');
  bubble.className = 'message-bubble';
  bubble.textContent = text;

  if (citation) {
    const citationEl = document.createElement('div');
    citationEl.className = 'citation';
    citationEl.textContent = citation;
    bubble.appendChild(citationEl);
  }

  row.appendChild(bubble);
  chatMessages.appendChild(row);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function setUploadStatus(message, type) {
  uploadStatus.textContent = message;
  uploadStatus.className = `status-box ${type}`;
  uploadStatus.classList.remove('hidden');
}

uploadForm.addEventListener('submit', async (event) => {
  event.preventDefault();

  const file = pdfInput.files[0];
  if (!file) {
    setUploadStatus('Please choose a PDF file first.', 'error');
    return;
  }

  if (file.type !== 'application/pdf' && !file.name.toLowerCase().endswith('.pdf')) {
    setUploadStatus('Only PDF files are allowed.', 'error');
    return;
  }

  const formData = new FormData();
  formData.append('file', file);

  uploadButton.disabled = true;
  uploadButton.textContent = 'Uploading...';
  setUploadStatus('Uploading and indexing PDF...', 'success');

  try {
    const response = await fetch('/upload', {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || 'Upload failed');
    }

   setUploadStatus(data.message, 'success');
    fileLabel.textContent = 'Choose a PDF to upload';
    pdfInput.value = '';
  } catch (error) {
    setUploadStatus(error.message || 'Could not upload the PDF.', 'error');
  } finally {
    uploadButton.disabled = false;
    uploadButton.textContent = 'Upload and Index';
  }
});

chatForm.addEventListener('submit', async (event) => {
  event.preventDefault();

  const question = questionInput.value.trim();
  if (!question) {
    return;
  }

  addMessage('user', question);
  questionInput.value = '';
  sendButton.disabled = true;
  sendButton.textContent = 'Sending...';
  addMessage('assistant', 'Thinking...', '');

  try {
    const response = await fetch('/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ question }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || 'Chat request failed');
    }

    const lastAssistantMessage = chatMessages.querySelectorAll('.message-row.assistant .message-bubble');
    const latest = lastAssistantMessage[lastAssistantMessage.length - 1];
    if (latest) {
      latest.textContent = data.answer;
      latest.innerHTML = data.answer.replace(/\n/g, '<br>');
    }
  } catch (error) {
    const lastAssistantMessage = chatMessages.querySelectorAll('.message-row.assistant .message-bubble');
    const latest = lastAssistantMessage[lastAssistantMessage.length - 1];
    if (latest) {
      latest.textContent = `Error: ${error.message || 'Unable to get an answer right now.'}`;
    }
  } finally {
    sendButton.disabled = false;
    sendButton.textContent = 'Send';
  }
});

questionInput.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    chatForm.requestSubmit();
  }
});
