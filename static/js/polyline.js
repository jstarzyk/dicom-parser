let canvas;
let ctx;

let canvas_;
let canvas__;

const distanceBox = {
    height: 20,
    offsetX: 10,
    offsetY: -30,
    background: "black",
    foreground: "white"
};
const canvasInfo = {
    paddingY: 2 * Math.abs(distanceBox.offsetY),
    paddingX: 9 * distanceBox.height,
};
const font = `${distanceBox.height}px sans-serif`;
const textBaseline = "bottom";
const strokeStyle = "red";

let points;
let currentDistance;
let followPointer;
let state;

function startPolyline() {
    // resetPolylineData();
    // clearCanvas();
    addListeners();
}

function resetPolyline() {
    removeListeners();
    clearCanvas();
    resetPolylineData();
}

function enableResizing() {
    onresize();
    window.addEventListener("resize", onresize);
}

function addListeners() {
    window.addEventListener("keyup", onkeyup);
    canvas.addEventListener("pointerdown", onpointerdown);
    canvas.addEventListener("pointerup", onpointerup);
    canvas.addEventListener("pointermove", onpointermove);
}

function removeListeners() {
    window.removeEventListener("keyup", onkeyup);
    canvas.removeEventListener("pointerdown", onpointerdown);
    canvas.removeEventListener("pointerup", onpointerup);
    canvas.removeEventListener("pointermove", onpointermove);
}

function resetPolylineData() {
    followPointer = false;
    currentDistance = 0;
    points = [];
    state = 2;
}

// function resizeCanvas(height = false, width = true) {
//     if (width) {
//         canvas.width += 2 * canvasInfo.paddingX;
//     }
//     if (height) {
//         canvas.height += 2 * canvasInfo.paddingY;
//     }
//     canvas_.style.minWidth = canvas.width + "px";
//     console.log('resized');
//     measure();
// }

function setupCanvas(img) {
    canvas.height = img.naturalHeight + 2 * canvasInfo.paddingY;
    canvas.width = img.naturalWidth + 2 * canvasInfo.paddingX;
    canvas_ = canvas.parentElement;
    canvas__ = canvas_.parentElement;
    canvas_.style.minWidth = canvas.width + "px";

    ctx.font = font;
    ctx.textBaseline = textBaseline;
    ctx.strokeStyle = strokeStyle;
}

function clearCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function getCoordinates(event) {
    return {x: event.offsetX, y: event.offsetY};
}

function drawPolyline() {
    if (points.length < 2) {
        return;
    }
    let point = points[0];
    ctx.beginPath();
    ctx.moveTo(point.x, point.y);
    for (let p = 1; p < points.length; p++) {
        point = points[p];
        ctx.lineTo(point.x, point.y);
    }
    ctx.stroke();
}

function measureDistance(p1, p2) {
    const dx = p2.x - p1.x;
    const dy = p2.y - p1.y;
    return (dx ** 2 + dy ** 2) ** 0.5;
}

function calculateDistance() {
    const p2 = points[points.length - 1];
    const p1 = points[points.length - 2];
    return p1 ? currentDistance + measureDistance(p1, p2) : 0;
}

// function fitDistance(point, width, height) {
//     let x = point.x + distanceBox.offsetX;
//     let y = point.y + distanceBox.offsetY;
//     if (x < 0) {
//         x = 0;
//     } else if (x + width > canvas.width) {
//         x = canvas.width - width;
//     }
//     if (y < 0) {
//         y = 0;
//     } else if (y + height > canvas.height) {
//         y = canvas.height - height;
//     }
//     return {x: x, y: y};
// }

function drawDistance(point, distance) {
    const text = Math.round(distance) + " px";
    const width = ctx.measureText(text).width;
    const height = distanceBox.height;
    const x = point.x + distanceBox.offsetX;
    const y = point.y + distanceBox.offsetY;
    // const {x, y} = fitDistance(point, width, height);
    ctx.fillStyle = distanceBox.background;
    ctx.fillRect(x, y, width, height);
    ctx.fillStyle = distanceBox.foreground;
    ctx.fillText(text, x, y + height);
}

function measure(point) {
    let distance;
    if (point) {
        distance = calculateDistance();
    } else {
        distance = currentDistance;
        point = points[points.length - 1];
    }
    clearCanvas();
    drawPolyline();
    drawDistance(point, distance);
    return distance;
}

function incorrectCoordinates(point) {
    return point.x < canvasInfo.paddingX
        || point.x > canvas.width - canvasInfo.paddingX
        || point.y < canvasInfo.paddingY
        || point.y > canvas.height - canvasInfo.paddingY
}

function onpointerup(event) {
    if (state === 2) {
        followPointer = true;
    }
}

function onpointerdown(event) {
    if (state === 2) {
        const point = getCoordinates(event);
        if (incorrectCoordinates(point)) {
            return;
        }
        followPointer = false;
        points.push(point);
        currentDistance = measure(point);
    }
}

function onpointermove(event) {
    if (state === 2 && followPointer) {
        const point = getCoordinates(event);
        if (incorrectCoordinates(point)) {
            return;
        }
        followPointer = false;
        points.push(point);
        measure(point);
        points.pop();
        followPointer = true;
    }
}

function onkeyup(event) {
    if (state > 0 && event.key === "Escape") {
        if (state === 2) {
            if (points.length > 0) {
                measure();
                state--;
            }
        } else if (state === 1) {
            resetPolyline();
        }
    }
}

function onresize() {
    canvas__.scrollTo((canvas_.offsetWidth - canvas__.offsetWidth) / 2, 0);
}
