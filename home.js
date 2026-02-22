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

// Выбор файла (используем метод коллеги)
selectBtn.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', function(event) {
    const file = event.target.files[0];
    if (file) {
        currentAudioFile = file;
        fileNameDisplay.textContent = '📄 ' + file.name;
        analyzeBtn.style.display = 'block';
        
        // Создаем временный URL для файла (метод коллеги)
        const fileURL = URL.createObjectURL(file);
        audioPlayer.src = fileURL;
        
        // Показываем общую длительность после загрузки метаданных
        audioPlayer.addEventListener('loadedmetadata', function() {
            totalTimeDisplay.textContent = formatTime(audioPlayer.duration);
        });
        
        // Автоматически воспроизводим (опционально)
        // audioPlayer.play();
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
    const progress = (audioPlayer.currentTime / audioPlayer.duration) * 100;
    spectrogramProgressBar.style.width = progress + '%';
    
    currentTimeDisplay.textContent = formatTime(audioPlayer.currentTime);
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

// Обработка (имитация)
processBtn.addEventListener('click', function() {
    processingStatus.style.display = 'block';
    
    let progress = 0;
    const interval = setInterval(() => {
        progress += 10;
        processingProgress.style.width = progress + '%';
        
        if (progress >= 100) {
            clearInterval(interval);
            setTimeout(() => {
                frameProcessing.style.display = 'none';
                
                // Случайный результат (для демо)
                if (Math.random() > 0.5) {
                    resultFake.style.display = 'block';
                } else {
                    resultReal.style.display = 'block';
                }
                frameResult.style.display = 'block';
                
                // Останавливаем воспроизведение
                audioPlayer.pause();
            }, 500);
        }
    }, 200);
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
    frameUpload.style.display = 'block';
    frameProcessing.style.display = 'none';
    frameResult.style.display = 'none';
    resultFake.style.display = 'none';
    resultReal.style.display = 'none';
    processingStatus.style.display = 'none';
    analyzeBtn.style.display = 'none';
    fileNameDisplay.textContent = '';
    fileInput.value = '';
    audioPlayer.src = '';
    audioPlayer.pause();
    spectrogramProgressBar.style.width = '0%';
    currentTimeDisplay.textContent = '0:00';
    totalTimeDisplay.textContent = '0:00';
});
