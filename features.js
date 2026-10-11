// Feature extractors replicating scikit-image on a 48x48 uint8 grayscale image (row-major).
export const SIZE = 48;

const pixel = (img, r, c) => (r < 0 || c < 0 || r >= SIZE || c >= SIZE ? 0 : img[r * SIZE + c]);

function bilinear(img, r, c) {
	const r0 = Math.floor(r), c0 = Math.floor(c), r1 = Math.ceil(r), c1 = Math.ceil(c);
	const dr = r - r0, dc = c - c0;
	const top = (1 - dc) * pixel(img, r0, c0) + dc * pixel(img, r0, c1);
	const bottom = (1 - dc) * pixel(img, r1, c0) + dc * pixel(img, r1, c1);
	return (1 - dr) * top + dr * bottom;
}

// local_binary_pattern(img, P, R, method='uniform') histogrammed over a grid x grid layout
export function lbp(img, P = 16, R = 2, grid = 6) {
	const round5 = x => Math.round(x * 1e5) / 1e5;
	const offsets = Array.from({ length: P }, (_, i) => [
		round5(-R * Math.sin(2 * Math.PI * i / P)),
		round5(R * Math.cos(2 * Math.PI * i / P)),
	]);
	const bins = P + 2, cell = SIZE / grid;
	const hist = new Float32Array(grid * grid * bins);

	for (let r = 0; r < SIZE; r++) for (let c = 0; c < SIZE; c++) {
		const bits = offsets.map(([dr, dc]) => (bilinear(img, r + dr, c + dc) - img[r * SIZE + c] >= 0 ? 1 : 0));
		let changes = 0;
		for (let i = 0; i < P - 1; i++) changes += bits[i] !== bits[i + 1];
		const code = changes <= 2 ? bits.reduce((a, b) => a + b) : P + 1;
		hist[(Math.floor(r / cell) * grid + Math.floor(c / cell)) * bins + code]++;
	}
	return hist;
}

const l2 = v => {
	const norm = Math.sqrt(v.reduce((sum, x) => sum + x * x, 0) + 1e-10);
	return v.map(x => x / norm);
};
const l2hys = v => l2(l2(v).map(x => Math.min(x, 0.2)));

// hog(img, orientations, pixels_per_cell=(cell, cell), cells_per_block=(block, block), block_norm='L2-Hys')
export function hog(img, orientations = 9, cell = 8, block = 2) {
	const n = SIZE / cell;
	const cells = new Float64Array(n * n * orientations);

	for (let r = 0; r < SIZE; r++) for (let c = 0; c < SIZE; c++) {
		const gr = r > 0 && r < SIZE - 1 ? img[(r + 1) * SIZE + c] - img[(r - 1) * SIZE + c] : 0;
		const gc = c > 0 && c < SIZE - 1 ? img[r * SIZE + c + 1] - img[r * SIZE + c - 1] : 0;
		let angle = (Math.atan2(gr, gc) * (180 / Math.PI)) % 180;
		if (angle < 0) angle += 180;
		const bin = Math.floor(angle / (180 / orientations));
		if (bin < orientations) cells[(Math.floor(r / cell) * n + Math.floor(c / cell)) * orientations + bin] += Math.hypot(gr, gc) / (cell * cell);
	}

	const features = [];
	for (let br = 0; br <= n - block; br++) for (let bc = 0; bc <= n - block; bc++) {
		const v = [];
		for (let r = br; r < br + block; r++) for (let c = bc; c < bc + block; c++) {
			const start = (r * n + c) * orientations;
			v.push(...cells.subarray(start, start + orientations));
		}
		features.push(...l2hys(v));
	}
	return Float32Array.from(features);
}
