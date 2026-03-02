// Элементы
const fileInput = document.getElementById('audioFileInput');
const selectBtn = document.getElementById('selectFileBtn');
const fileNameDisplay = document.getElementById('fileNameDisplay');
const analyzeBtn = document.getElementById('analyzeBtn');
const frameUpload = document.getElementById('frameUpload');
const frameProcessing = document.getElementById('frameProcessing');
const frameResult = document.getElementById('frameResult');
const resultFake = document.getElementById('resultFake');
const resultReal = document.getElementById('resultReal');
const audioPlayer = document.getElementById('audioPlayer');
const processBtn = document.getElementById('processBtn');
const processingStatus = document.getElementById('processingStatus');
const processingProgress = document.getElementById('processingProgress');
const spectrogramIcon = document.getElementById('spectrogramIcon');
const spectrogramContainer = document.getElementById('spectrogramContainer');
const spectrogramProgressBar = document.getElementById('spectrogramProgressBar');
const homeNavBtn = document.getElementById('homeNavBtn');
const currentTimeDisplay = document.getElementById('currentTimeDisplay');
const totalTimeDisplay = document.getElementById('totalTimeDisplay');

let currentAudioFile = null;
let currentAudioUrl = null;

// Загружаем выбранную модель из настроек (localStorage)
let modelType = localStorage.getItem('selectedModel') || 'cnn';
console.log('✅ Используется модель:', modelType);

// Выбор файла
selectBtn.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', function(event) {
    const file = event.target.files[0];
    if (file) {
        currentAudioFile = file;
        fileNameDisplay.textContent = '📄 ' + file.name;
        analyzeBtn.style.display = 'block';

        // Очищаем предыдущий URL
        if (currentAudioUrl) {
            URL.revokeObjectURL(currentAudioUrl);
        }

        // Создаем временный URL для файла
        currentAudioUrl = URL.createObjectURL(file);
        audioPlayer.src = currentAudioUrl;

        // Показываем общую длительность после загрузки метаданных
        audioPlayer.addEventListener('loadedmetadata', function() {
            totalTimeDisplay.textContent = formatTime(audioPlayer.duration);
        });
    }
});

// Анализ (переход к фрейму обработки)
analyzeBtn.addEventListener('click', function() {
    frameUpload.style.display = 'none';
    frameProcessing.style.display = 'block';

    // Запускаем воспроизведение при переходе
    audioPlayer.play();
});

// Обновление прогресса на спектрограмме
audioPlayer.addEventListener('timeupdate', function() {
    if (audioPlayer.duration) {
        const progress = (audioPlayer.currentTime / audioPlayer.duration) * 100;
        spectrogramProgressBar.style.width = progress + '%';
        currentTimeDisplay.textContent = formatTime(audioPlayer.currentTime);
    }
});

audioPlayer.addEventListener('ended', function() {
    spectrogramProgressBar.style.width = '0%';
    currentTimeDisplay.textContent = '0:00';
});

// Клик по спектрограмме для перемотки
spectrogramContainer.addEventListener('click', function(e) {
    const rect = spectrogramContainer.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const percentage = clickX / rect.width;
    const newTime = percentage * audioPlayer.duration;

    if (!isNaN(newTime) && isFinite(newTime)) {
        audioPlayer.currentTime = newTime;
    }
});

// ===== ОТПРАВКА ЗАПРОСА НА СЕРВЕР (ОБНОВЛЕННЫЙ FETCH) =====
processBtn.addEventListener('click', function() {
    if (!currentAudioFile) {
        alert('❌ Сначала выберите аудиофайл');
        return;
    }

    // Показываем статус обработки
    processingStatus.style.display = 'block';
    processingProgress.style.width = '0%';

    // Создаем FormData для отправки файла
    const formData = new FormData();
    formData.append('file', currentAudioFile);
    formData.append('model_type', modelType);

    console.log('🚀 Отправка файла на сервер:', currentAudioFile.name);
    console.log('🤖 Модель:', modelType);

    // Имитация прогресса загрузки (визуализация)
    let progress = 0;
    const progressInterval = setInterval(() => {
        progress += 5;
        if (progress <= 90) {
            processingProgress.style.width = progress + '%';
        }
    }, 100);

    // FETCH запрос к API с явным указанием CORS
    fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: formData,
        mode: 'cors',
        credentials: 'same-origin'
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        clearInterval(progressInterval);

        if (data.success) {
            processingProgress.style.width = '100%';

            console.log('✅ Ответ сервера:', data);

            // Небольшая задержка для плавности
            setTimeout(() => {
                // Скрываем фрейм обработки
                frameProcessing.style.display = 'none';

                // Удаляем старый confidence если был
                const oldConfidence = document.querySelector('.confidence-text');
                if (oldConfidence) oldConfidence.remove();

                // Выбираем нужный фрейм результата
                if (data.prediction === 'fake') {
                    resultFake.style.display = 'block';
                    resultReal.style.display = 'none';

                    // Добавляем информацию об уверенности
                    const confidenceEl = document.createElement('p');
                    confidenceEl.className = 'confidence-text';
                    confidenceEl.textContent = `Уверенность: ${(data.confidence * 100).toFixed(1)}% | Модель: ${data.model_type || modelType}`;
                    resultFake.appendChild(confidenceEl);
                } else {
                    resultReal.style.display = 'block';
                    resultFake.style.display = 'none';

                    // Добавляем информацию об уверенности
                    const confidenceEl = document.createElement('p');
                    confidenceEl.className = 'confidence-text';
                    confidenceEl.textContent = `Уверенность: ${(data.confidence * 100).toFixed(1)}% | Модель: ${data.model_type || modelType}`;
                    resultReal.appendChild(confidenceEl);
                }

                frameResult.style.display = 'block';

                // Останавливаем воспроизведение
                audioPlayer.pause();
            }, 500);
        } else {
            throw new Error(data.error || 'Неизвестная ошибка');
        }
    })
    .catch(error => {
        clearInterval(progressInterval);
        processingStatus.style.display = 'none';
        alert('❌ Ошибка при обработке: ' + error.message);
        console.error('❌ Fetch error:', error);
    });
});

// Форматирование времени
function formatTime(seconds) {
    if (isNaN(seconds)) return '0:00';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return mins + ':' + (secs < 10 ? '0' : '') + secs;
}

// Возврат на главную
homeNavBtn.addEventListener('click', function() {
    // Очищаем все
    frameUpload.style.display = 'block';
    frameProcessing.style.display = 'none';
    frameResult.style.display = 'none';
    resultFake.style.display = 'none';
    resultReal.style.display = 'none';
    processingStatus.style.display = 'none';
    analyzeBtn.style.display = 'none';

    // Очищаем файл
    fileNameDisplay.textContent = '';
    fileInput.value = '';
    audioPlayer.pause();
    audioPlayer.src = '';
    if (currentAudioUrl) {
        URL.revokeObjectURL(currentAudioUrl);
        currentAudioUrl = null;
    }

    // Сбрасываем прогресс
    spectrogramProgressBar.style.width = '0%';
    currentTimeDisplay.textContent = '0:00';
    totalTimeDisplay.textContent = '0:00';

    // Удаляем добавленные элементы confidence
    const confidenceEls = document.querySelectorAll('.confidence-text');
    confidenceEls.forEach(el => el.remove());
});

// Получение информации об устройстве (опционально)
fetch('http://localhost:5000/api/device')
    .then(response => response.json())
    .then(data => {
        console.log('ℹ️ Device info:', data);
    })
    .catch(err => console.log('Device info not available'));

// Следим за изменениями модели в настройках
window.addEventListener('storage', function(e) {
    if (e.key === 'selectedModel') {
        modelType = e.newValue;
        console.log('🔄 Модель обновлена из настроек:', modelType);
    }
});