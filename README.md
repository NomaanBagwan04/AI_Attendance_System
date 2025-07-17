# 📅 AI Attendance System<br>

An AI-powered attendance system that uses **face detection and recognition** to automate attendance tracking. <br>
Built with **OpenCV**, **MTCNN**, and **FaceNet**, it captures faces from a live camera feed and logs attendance in real time.<br>

\---<br>

## ✨ Features<br>

* 📹 Real-time face detection and recognition<br>
* ✅ Automatic attendance logging with timestamps<br>
* 📲 Webcam support for live monitoring<br>
* ⚙ Easy registration of new faces<br>
* 🔐 CSV-based attendance logs for offline records<br>

\---<br>

## 📊 Tech Stack<br>

* **Python 3.8+**<br>
* **OpenCV** – image capture & preprocessing<br>
* **MTCNN** – for face detection<br>
* **FaceNet** – for face embedding and comparison<br>
* **NumPy, Pandas** – data handling<br>

\---<br>

## 📦 Installation<br>

Clone the repository:<br>

```bash
git clone https://github.com/NomaanBagwan04/AI_Attendance_System.git
cd AI_Attendance_System
```

Install the dependencies:<br>

```bash
pip install -r requirements.txt
```

\---<br>

## ▶️ Usage<br>

1. Register known faces by adding images to the `images/` folder (one folder per person).<br>
2. Run the main script:<br>

```bash
python main.py
```

3. The system will:<br>

   * Open webcam<br>
   * Detect and recognize faces<br>
   * Log attendance to `Attendance.csv`<br>

\---<br>

## 📁 Project Structure<br>

```
.
├── images/             # Folder containing subfolders of known individuals
├── main.py             # Main attendance script
├── attendance.csv      # Output log file
├── facenet_keras.h5    # Pretrained FaceNet model
├── requirements.txt
└── README.md
```

\---<br>

## 🚀 Future Improvements<br>

* [ ] GUI for registration and live feed display<br>
* [ ] Database integration (SQLite/MySQL)<br>
* [ ] Notification system for alerts<br>
* [ ] Cloud deployment support<br>

\---<br>

## 🙌 Acknowledgements<br>

* [MTCNN](https://github.com/ipazc/mtcnn)<br>
* [FaceNet Keras](https://github.com/nyoki-mtl/keras-facenet)<br>
* [OpenCV](https://opencv.org/)<br>

\---<br>

