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

const colors = ['bg-primary', 'bg-warning', 'bg-info'];

$('models').innerHTML = models.map((m, i) => `<span class="badge ${colors[i]} me-1">&nbsp;</span>${m.name}`).join('<br>');

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

const escape = text => String(text).replace(/[&<>"]/g, ch => `&#${ch.charCodeAt(0)};`);

async function runModel(model) {
	try {
		const session = await sessions[model.file];
		const start = performance.now();
		const outputs = await session.run({ [session.inputNames[0]]: extractors[model.features](image) });
		const scores = Array.from(Object.values(outputs).find(t => t.type === 'float32').data);
		const probabilities = model.softmax ? softmax(scores) : scores;
		const best = probabilities.indexOf(Math.max(...probabilities));
		return { probabilities, best, time: performance.now() - start };
	} catch (error) {
		delete sessions[model.file];
		return { error: error.message ?? error };
	}
}

function consensus(results) {
	const votes = {};
	results.filter(r => !r.error).forEach(r => votes[r.best] = (votes[r.best] ?? 0) + 1);
	const [best, count] = Object.entries(votes).sort((a, b) => b[1] - a[1])[0] ?? [];
	if (!count) return '<div class="alert alert-danger">All models failed.</div>';
	if (count === 1 && Object.keys(votes).length > 1) return '<h2 class="h5 mb-3">The models disagree</h2>';
	return `<h2 class="h5 mb-3">You look: ${labels[best]} <small class="text-body-secondary">(${count}/${models.length} models agree)</small></h2>`;
}

function showResults(results) {
	const rows = models.map((model, i) => {
		const r = results[i];
		const name = `<span class="badge ${colors[i]} me-1">&nbsp;</span>${model.name}`;
		return r.error
			? `<tr><td>${name}</td><td colspan="3" class="text-danger">${escape(r.error)}</td></tr>`
			: `<tr><td>${name}</td><td class="fw-semibold">${labels[r.best]}</td>
				<td class="text-end">${(r.probabilities[r.best] * 100).toFixed(1)}%</td>
				<td class="text-end">${r.time.toFixed(0)} ms</td></tr>`;
	}).join('');

	const bars = labels.map((label, l) => `
		<div class="d-flex align-items-center gap-2 mb-2 small">
			<span style="width: 5rem">${label}</span>
			<div class="flex-grow-1">${models.map((model, i) => {
				const p = results[i].error ? 0 : results[i].probabilities[l] * 100;
				return `<div class="d-flex align-items-center gap-2" title="${model.name}: ${p.toFixed(1)}%">
					<div class="progress flex-grow-1" style="height: 6px"><div class="progress-bar ${colors[i]}" style="width: ${p}%"></div></div>
					<span class="text-end text-body-secondary" style="width: 3rem; font-size: .7rem">${p.toFixed(1)}%</span>
				</div>`;
			}).join('')}</div>
		</div>`).join('');

	$('result').innerHTML = consensus(results) + `
		<div class="table-responsive"><table class="table table-sm small align-middle">
			<thead><tr><th>Model</th><th>Prediction</th><th class="text-end">Confidence</th><th class="text-end">Inference</th></tr></thead>
			<tbody>${rows}</tbody>
		</table></div>
		<h3 class="h6 mt-2 mb-3">Probabilities per emotion</h3>${bars}`;
}

$('predict').onclick = async () => {
	$('predict').disabled = true;
	$('predict').textContent = 'Running…';
	// Models load in parallel, but inference runs one at a time so the WASM backend is not shared and timings are fair
	models.forEach(model => sessions[model.file] ??= loadSession(model.file));
	const results = [];
	for (const model of models) results.push(await runModel(model));
	showResults(results);
	$('result').hidden = false;
	$('predict').disabled = false;
	$('predict').textContent = 'Predict with all models';
};
