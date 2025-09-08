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

async function setupCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
    });
    video.srcObject = stream;

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
    await setupCamera();
    video.play();
    await loadModel();
    detectPose();
}

main();
