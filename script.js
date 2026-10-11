import brotli from 'https://cdn.jsdelivr.net/npm/brotli-wasm@3.0.1/index.web.js';
import { SIZE, lbp, hog } from './features.js';

const $ = id => document.getElementById(id);
const { labels, models } = await (await fetch('config.json')).json();
const sessions = {};
let image = null; // 48x48 grayscale Uint8Array

const extractors = {
	pixels: img => new ort.Tensor('float32', Float32Array.from(img, v => v / 255), [1, SIZE, SIZE, 1]),
	lbp: img => new ort.Tensor('float32', lbp(img), [1, 648]),
	hog: img => new ort.Tensor('float32', hog(img), [1, 900]),
};

$('model').innerHTML = models.map((m, i) => `<option value="${i}">${m.name}</option>`).join('');

// Center-crops the source, resizes it to 48x48 and converts to grayscale (OpenCV weights)
function setImage(source, width, height) {
	const side = Math.min(width, height);
	const ctx = $('preview').getContext('2d', { willReadFrequently: true });
	ctx.drawImage(source, (width - side) / 2, (height - side) / 2, side, side, 0, 0, SIZE, SIZE);
	const pixels = ctx.getImageData(0, 0, SIZE, SIZE);
	const { data } = pixels;
	image = new Uint8Array(SIZE * SIZE);
	for (let i = 0; i < image.length; i++) {
		image[i] = Math.round(0.299 * data[4 * i] + 0.587 * data[4 * i + 1] + 0.114 * data[4 * i + 2]);
		data[4 * i] = data[4 * i + 1] = data[4 * i + 2] = image[i];
	}
	ctx.putImageData(pixels, 0, 0);
	$('preview').hidden = false;
	$('predict').disabled = false;
}

$('file').onchange = async ({ target }) => {
	const bitmap = await createImageBitmap(target.files[0]);
	setImage(bitmap, bitmap.width, bitmap.height);
};

$('camera').onclick = async () => {
	$('video').srcObject = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } });
	$('video').hidden = $('capture').hidden = false;
};

$('capture').onclick = () => {
	const video = $('video');
	setImage(video, video.videoWidth, video.videoHeight);
	video.srcObject.getTracks().forEach(track => track.stop());
	video.hidden = $('capture').hidden = true;
};

async function loadSession(file) {
	let bytes = new Uint8Array(await (await fetch(file)).arrayBuffer());
	if (file.endsWith('.br')) bytes = (await brotli).decompress(bytes);
	return ort.InferenceSession.create(bytes);
}

const softmax = scores => {
	const exps = scores.map(s => Math.exp(s - Math.max(...scores)));
	return exps.map(e => e / exps.reduce((a, b) => a + b));
};

function showResult(probabilities) {
	const ranked = labels.map((label, i) => ({ label, p: probabilities[i] * 100 })).sort((a, b) => b.p - a.p);
	$('result').innerHTML = `<h2 class="h5 mb-3">You look: ${ranked[0].label}</h2>` + ranked.map(({ label, p }) => `
		<div class="d-flex align-items-center gap-2 mb-1 small">
			<span style="width: 5rem">${label}</span>
			<div class="progress flex-grow-1"><div class="progress-bar" style="width: ${p}%"></div></div>
			<span class="text-end" style="width: 3.5rem">${p.toFixed(1)}%</span>
		</div>`).join('');
}

$('predict').onclick = async () => {
	const model = models[$('model').value];
	$('predict').disabled = true;
	$('predict').textContent = 'Running…';
	try {
		const session = await (sessions[model.file] ??= loadSession(model.file));
		const outputs = await session.run({ [session.inputNames[0]]: extractors[model.features](image) });
		const scores = Array.from(Object.values(outputs).find(t => t.type === 'float32').data);
		showResult(model.softmax ? softmax(scores) : scores);
	} catch (error) {
		delete sessions[model.file];
		$('result').innerHTML = `<div class="alert alert-danger mb-0">${error.message}</div>`;
	}
	$('result').hidden = false;
	$('predict').disabled = false;
	$('predict').textContent = 'Predict emotion';
};
