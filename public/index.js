import * as poseDetection from '@tensorflow-models/pose-detection';
import * as tf from '@tensorflow/tfjs';

let detector;
let poses;
let pushupCount = 0;
let pushupState = 'up'; // 'up' or 'down'

const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const counter = document.getElementById('counter');
const cameraSelect = document.getElementById('camera-select');
const exportButton = document.getElementById('export-button');
const clearButton = document.getElementById('clear-button');
const historyTableBody = document.querySelector('#history-table tbody');

let currentStream;

function loadHistory() {
    const history = JSON.parse(localStorage.getItem('pushupHistory')) || [];
    historyTableBody.innerHTML = '';
    history.forEach((rep, index) => {
        const row = historyTableBody.insertRow();
        const cell1 = row.insertCell(0);
        const cell2 = row.insertCell(1);
        cell1.innerText = index + 1;
        cell2.innerText = new Date(rep.timestamp).toLocaleString();
    });
}

function saveRepetition() {
    const history = JSON.parse(localStorage.getItem('pushupHistory')) || [];
    history.push({ timestamp: new Date().toISOString() });
    localStorage.setItem('pushupHistory', JSON.stringify(history));
    loadHistory();
}


async function getCameras() {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter(device => device.kind === 'videoinput');
    videoDevices.forEach(device => {
        const option = document.createElement('option');
        option.value = device.deviceId;
        option.text = device.label || `Camera ${cameraSelect.length + 1}`;
        cameraSelect.appendChild(option);
    });
}

async function setupCamera(deviceId) {
    if (currentStream) {
        currentStream.getTracks().forEach(track => {
            track.stop();
        });
    }
    const stream = await navigator.mediaDevices.getUserMedia({
        video: { deviceId: deviceId ? { exact: deviceId } : undefined },
    });
    video.srcObject = stream;
    currentStream = stream;

    return new Promise((resolve) => {
        video.onloadedmetadata = () => {
            resolve(video);
        };
    });
}

async function loadModel() {
    const model = poseDetection.SupportedModels.MoveNet;
    detector = await poseDetection.createDetector(model);
}

function isPushupDown(keypoints) {
    const leftShoulder = keypoints.find(k => k.name === 'left_shoulder');
    const leftElbow = keypoints.find(k => k.name === 'left_elbow');
    const rightShoulder = keypoints.find(k => k.name === 'right_shoulder');
    const rightElbow = keypoints.find(k => k.name === 'right_elbow');

    if (leftShoulder && leftElbow && rightShoulder && rightElbow) {
        const leftElbowAngle = Math.atan2(leftElbow.y - leftShoulder.y, leftElbow.x - leftShoulder.x) * 180 / Math.PI;
        const rightElbowAngle = Math.atan2(rightElbow.y - rightShoulder.y, rightElbow.x - rightShoulder.x) * 180 / Math.PI;
        return leftElbowAngle < 100 && rightElbowAngle < 100;
    }
    return false;
}

function isPushupUp(keypoints) {
    const leftShoulder = keypoints.find(k => k.name === 'left_shoulder');
    const leftElbow = keypoints.find(k => k.name === 'left_elbow');
    const rightShoulder = keypoints.find(k => k.name === 'right_shoulder');
    const rightElbow = keypoints.find(k => k.name === 'right_elbow');

    if (leftShoulder && leftElbow && rightShoulder && rightElbow) {
        const leftElbowAngle = Math.atan2(leftElbow.y - leftShoulder.y, leftElbow.x - leftShoulder.x) * 180 / Math.PI;
        const rightElbowAngle = Math.atan2(rightElbow.y - rightShoulder.y, rightElbow.x - rightShoulder.x) * 180 / Math.PI;
        return leftElbowAngle > 160 && rightElbowAngle > 160;
    }
    return false;
}


async function detectPose() {
    poses = await detector.estimatePoses(video);

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (poses && poses.length > 0) {
        const keypoints = poses[0].keypoints;

        if (pushupState === 'up' && isPushupDown(keypoints)) {
            pushupState = 'down';
        } else if (pushupState === 'down' && isPushupUp(keypoints)) {
            pushupState = 'up';
            pushupCount++;
            counter.innerText = `Push-ups: ${pushupCount}`;
            saveRepetition();
        }


        // Draw skeleton
        for (const keypoint of keypoints) {
            if (keypoint.score > 0.5) {
                ctx.beginPath();
                ctx.arc(keypoint.x, keypoint.y, 5, 0, 2 * Math.PI);
                ctx.fillStyle = 'red';
                ctx.fill();
            }
        }

        const adjacentPairs = poseDetection.util.getAdjacentPairs(poseDetection.SupportedModels.MoveNet);
        for (const pair of adjacentPairs) {
            const from = keypoints.find(k => k.name === pair[0]);
            const to = keypoints.find(k => k.name === pair[1]);

            if (from && to && from.score > 0.5 && to.score > 0.5) {
                ctx.beginPath();
                ctx.moveTo(from.x, from.y);
                ctx.lineTo(to.x, to.y);
                ctx.strokeStyle = 'green';
                ctx.lineWidth = 2;
                ctx.stroke();
            }
        }


    }

    requestAnimationFrame(detectPose);
}

async function main() {
    await tf.ready();
    await getCameras();
    await setupCamera(cameraSelect.value);
    video.play();
    await loadModel();
    detectPose();

    cameraSelect.addEventListener('change', async () => {
        await setupCamera(cameraSelect.value);
        video.play();
    });

    exportButton.addEventListener('click', () => {
        const history = JSON.parse(localStorage.getItem('pushupHistory')) || [];
        if (history.length === 0) {
            alert('No history to export.');
            return;
        }
        let csvContent = 'data:text/csv;charset=utf-8,Repetition,Timestamp\n';
        history.forEach((rep, index) => {
            csvContent += `${index + 1},${new Date(rep.timestamp).toLocaleString()}\n`;
        });
        const encodedUri = encodeURI(csvContent);
        const link = document.createElement('a');
        link.setAttribute('href', encodedUri);
        link.setAttribute('download', 'pushup_history.csv');
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    });

    clearButton.addEventListener('click', () => {
        if (confirm('Are you sure you want to clear the history?')) {
            localStorage.removeItem('pushupHistory');
            pushupCount = 0;
            counter.innerText = 'Push-ups: 0';
            loadHistory();
        }
    });

    loadHistory();
}

main();
