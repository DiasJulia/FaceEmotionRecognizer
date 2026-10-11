import { SIZE } from './features.js';

const $ = id => document.getElementById(id);
const { models } = await (await fetch('config.json')).json();
let image = null; // 48x48 grayscale Uint8Array

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
