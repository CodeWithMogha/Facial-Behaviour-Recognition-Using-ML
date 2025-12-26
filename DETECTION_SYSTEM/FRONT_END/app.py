from flask import Flask, render_template, Response, redirect, jsonify
from camera import VideoCamera
import sqlite3
from collections import Counter

app = Flask(__name__)
camera = None


@app.route('/')
def index():
    global camera
    if not camera:
        camera = VideoCamera()
    return render_template('index.html', camera_on=True)


@app.route('/video_feed')
def video_feed():
    global camera
    def gen(camera):
        while True:
            frame = camera.get_frame()
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    return Response(gen(camera), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stop_feed')
def stop_feed():
    global camera
    if camera:
        del camera
        camera = None
    return redirect('/')


@app.route('/get_emotion')
def get_emotion():
    global camera
    if camera:
        return jsonify({'emotion': camera.emotion})
    return jsonify({'emotion': "Neutral"})


@app.route('/get_user_log')
def get_user_log():
    conn = sqlite3.connect("user_logs.db")
    c = conn.cursor()
    c.execute("SELECT name, emotion, time, day, date FROM logs ORDER BY id DESC LIMIT 10")
    logs = c.fetchall()
    conn.close()

    return jsonify([
        {"name": row[0], "emotion": row[1], "time": row[2], "day": row[3], "date": row[4]}
        for row in logs
    ])


@app.route('/emotion_stats')
def emotion_stats():
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    try:
        conn = sqlite3.connect("user_logs.db")
        c = conn.cursor()
        c.execute("SELECT emotion FROM logs ORDER BY id DESC LIMIT 10")
        data = c.fetchall()
        conn.close()

        # Flatten and count emotions
        recent_emotions = [row[0] for row in data]
        emotion_counter = Counter(recent_emotions)
        total = sum(emotion_counter.values())

        percentages = {emotion: (emotion_counter.get(emotion, 0) / total) * 100 if total else 0
                       for emotion in emotions}

        return jsonify(percentages)

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(debug=True)
