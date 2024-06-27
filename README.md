# Automated Attendance System

## Overview

This project implements an automated attendance system using face recognition. The system captures video from a webcam, detects faces, matches them against known faces, and records attendance in a CSV file.

## Features

- Face detection and recognition using `face_recognition` library.
- Real-time video capture and processing with `opencv-python`.
- Attendance logging in a CSV file (`attendance.csv`).
- Serialization of face encodings using `pickle`.

## Installation

### Prerequisites

- Python 3.8 installed on your system.

### Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd automated-attendance-system

2. Install required packages:
    ```bash
    pip install -r requirements.txt

### Usage

1. Ensure your webcam is connected and accessible.

2. Run the main script
    ```bash
    python src/main.py
    
3. The application will start capturing video from the webcam. Recognized faces will be highlighted with their names displayed on the video feed. Press q to exit.

4. View attendance records in data/attendance.csv.

### Official Documentation

[Download PDF](/Automated_Attendance_System.pdf)
