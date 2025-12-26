const ctx = document.getElementById('emotionChart').getContext('2d');

const emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'];
const emojis = ['😠', '🤢', '😱', '😄', '😢', '😲', '😐'];

const barColors = [
  'rgba(107, 144, 128, 0.8)',
  'rgba(107, 144, 128, 0.7)',
  'rgba(107, 144, 128, 0.6)',
  'rgba(107, 144, 128, 0.5)',
  'rgba(107, 144, 128, 0.4)',
  'rgba(107, 144, 128, 0.3)',
  'rgba(107, 144, 128, 0.2)'
];

// ✅ Plugin to draw emojis and % on bars
const emojiAndPercentPlugin = {
  id: 'emojiAndPercentPlugin',
  afterDatasetsDraw(chart) {
    const { ctx, data, chartArea: { top, bottom, left, right }, scales: { x, y } } = chart;
    ctx.save();
    ctx.textAlign = 'center';
    ctx.fillStyle = '#f6fff8';
    ctx.font = '20px Arial';

    data.datasets[0].data.forEach((value, index) => {
      const xPos = x.getPixelForValue(index);
      const yPos = y.getPixelForValue(value);
      ctx.fillText(emojis[index], xPos, yPos - 10);
      ctx.font = '14px Arial';
      ctx.textBaseline = 'middle';
      const barTop = y.getPixelForValue(value);
      const barBottom = y.getPixelForValue(0);
      const barHeight = barBottom - barTop;
      ctx.fillText(`${value.toFixed(1)}%`, xPos, barTop + barHeight / 2);
    });

    ctx.restore();
  }
};

const emotionChart = new Chart(ctx, {
  type: 'bar',
  data: {
    labels: emotions,
    datasets: [{
      label: 'Emotion Percentage',
      data: emotions.map(() => 0),
      backgroundColor: barColors,
      borderRadius: 6,
      borderSkipped: false,
    }]
  },
  options: {
    responsive: true,
    animation: { duration: 500 },
    scales: {
      y: {
        beginAtZero: true,
        max: 100,
        ticks: {
          color: '#f6fff8',
          stepSize: 20,
          callback: val => `${val}%`
        },
        grid: { color: 'rgba(255,255,255,0.1)' }
      },
      x: {
        ticks: { color: '#f6fff8', font: { size: 14 } },
        grid: { display: false }
      }
    },
    plugins: {
      legend: { display: false },
      tooltip: {
        enabled: true,
        callbacks: {
          label: context => `${context.parsed.y.toFixed(1)}%`
        }
      }
    }
  },
  plugins: [emojiAndPercentPlugin]
});

let lastSpokenEmotion = null;

// ✅ Update chart using DB every 3 sec
function updateEmotionDataFromDB() {
  fetch('/emotion_stats')
    .then(res => res.json())
    .then(percentages => {
      const data = emotions.map(em => percentages[em] || 0);
      emotionChart.data.datasets[0].data = data;
      emotionChart.update();

      // Speak & update text box
      const maxVal = Math.max(...data);
      const maxIndex = data.indexOf(maxVal);
      const dominantEmotion = emotions[maxIndex];
      const dominantEmoji = emojis[maxIndex];

      const predictionText = document.getElementById('predictionText');
      predictionText.textContent = `You look ${dominantEmotion} ${dominantEmoji}`;

      const box = document.querySelector('.prediction-box');
      switch (dominantEmotion) {
        case 'Happy': box.style.backgroundColor = '#a4c3b2'; break;
        case 'Sad': box.style.backgroundColor = '#9a8c98'; break;
        case 'Angry': box.style.backgroundColor = '#e76f51'; break;
        case 'Surprise': box.style.backgroundColor = '#f4a261'; break;
        case 'Fear': box.style.backgroundColor = '#264653'; break;
        case 'Disgust': box.style.backgroundColor = '#2a9d8f'; break;
        case 'Neutral': box.style.backgroundColor = '#cce3de'; break;
        default: box.style.backgroundColor = '#a4c3b2';
      }

      // Speak if new emotion
      if (dominantEmotion !== lastSpokenEmotion) {
        speakEmotion(dominantEmotion);
        lastSpokenEmotion = dominantEmotion;
      }
    })
    .catch(err => console.error("Error fetching emotion stats:", err));
}

// ✅ Speak function
function speakEmotion(emotion) {
  const msg = new SpeechSynthesisUtterance(`You look ${emotion}`);
  msg.lang = 'en-US';
  msg.rate = 1;
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(msg);
}

// ✅ Run update every 3 seconds
setInterval(updateEmotionDataFromDB, 3000);

// Initial fetch
updateEmotionDataFromDB();
