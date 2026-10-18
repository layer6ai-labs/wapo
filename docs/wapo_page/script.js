'use strict';

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const svgNS = 'http://www.w3.org/2000/svg';
const colors = { peak: 'var(--chart-peak)', valley: 'var(--chart-valley)', ink: 'var(--ink)', teal: 'var(--accent)', muted: 'var(--muted)', line: 'var(--line)', boundary: 'var(--accent)', selected: 'var(--chart-selected)' };
function svgElement(tag, attributes = {}, value) {
  const element = document.createElementNS(svgNS, tag);
  Object.entries(attributes).forEach(([key, val]) => element.setAttribute(key, val));
  if (value !== undefined) element.textContent = value;
  return element;
}
function svgText(parent, x, y, value, attributes = {}) {
  const text = svgElement('text', { x, y, fill: colors.muted, 'font-family': 'Arial, sans-serif', 'font-size': 11, ...attributes }, value);
  parent.append(text);
  return text;
}
function updateProgress() {
  const extent = document.documentElement.scrollHeight - innerHeight;
  $('#progress').style.width = `${extent > 0 ? Math.min(100, 100 * scrollY / extent) : 0}%`;
}
addEventListener('scroll', updateProgress, { passive: true });
addEventListener('resize', updateProgress);
addEventListener('load', updateProgress);
const sections = $$('#overview, #background, .chapter');
if ('IntersectionObserver' in window) {
  const observer = new IntersectionObserver(entries => {
    const visible = entries.filter(entry => entry.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
    if (visible.length) {
      $$('.story-nav a').forEach(link => {
        const active = link.hash === `#${visible[0].target.id}`;
        link.classList.toggle('active', active);
        if (active) link.setAttribute('aria-current', 'location');
        else link.removeAttribute('aria-current');
      });
    }
  }, { rootMargin: '-8% 0px -55% 0px' });
  sections.forEach(section => observer.observe(section));
}
$$('details').forEach(details => details.addEventListener('toggle', updateProgress));

// Direct softmax update, independent of all measured training curves.
const toyDistributions = {
  concentrated: [.78, .10, .05, .03, .039, .001],
  spread: [.35, .28, .22, .08, .069, .001]
};
let sampled = 0;
function entropy(p) { return -p.reduce((total, v) => total + (v > 0 ? v * Math.log(v) : 0), 0); }
function softmax(logits) {
  const maximum = Math.max(...logits);
  const exp = logits.map(value => Math.exp(value - maximum));
  const denominator = exp.reduce((a, b) => a + b, 0);
  return exp.map(value => value / denominator);
}
function drawDistribution() {
  const p = toyDistributions[$('#distribution').value];
  const advantage = Number($('#advantage').value);
  const boundary = p.reduce((total, value) => total + value * value, 0);
  const updated = softmax(p.map((value, index) => Math.log(value) + .1 * advantage * ((index === sampled ? 1 : 0) - value)));
  const hChange = entropy(updated) - entropy(p);
  const tinyChange = updated[5] - p[5];
  const width = 650, height = 235, left = 36, top = 25, bottom = 189, scale = (bottom - top) / .85;
  const chart = svgElement('svg', { viewBox: `0 0 ${width} ${height}`, role: 'group', 'aria-label': 'Select a sampled token to compare the probabilities before and after one direct logit update.' });
  chart.append(svgElement('line', { x1: left, x2: 624, y1: bottom, y2: bottom, stroke: colors.line }));
  const boundaryY = bottom - boundary * scale;
  chart.append(svgElement('line', { x1: left, x2: 624, y1: boundaryY, y2: boundaryY, stroke: colors.boundary, 'stroke-dasharray': '4 4' }));
  svgText(chart, 623, Math.max(15, boundaryY - 7), `C(p) = ${boundary.toFixed(3)}`, { 'text-anchor': 'end', fill: colors.boundary });
  svgText(chart, 4, 16, 'p');
  p.forEach((value, index) => {
    const x = 55 + index * 92;
    const y = bottom - value * scale;
    const peak = value >= boundary;
    const token = svgElement('g', { role: 'button', tabindex: 0, 'aria-pressed': index === sampled ? 'true' : 'false', 'aria-label': `Token ${index + 1}, probability ${value}, ${peak ? 'peak' : 'valley'}${index === sampled ? ', sampled' : ''}` });
    token.append(svgElement('rect', { x: x - 8, y: top - 8, width: 66, height: bottom - top + 46, fill: index === sampled ? colors.selected : 'transparent', rx: 3, class: 'bar-focus' }));
    token.append(svgElement('rect', { x, y, width: 48, height: Math.max(1, value * scale), fill: peak ? colors.peak : colors.valley, class: 'bar-fill', opacity: .82 }));
    token.append(svgElement('rect', { x, y: bottom - updated[index] * scale, width: 48, height: Math.max(1, updated[index] * scale), fill: 'none', stroke: colors.ink, 'stroke-width': 1.4 }));
    svgText(token, x + 24, Math.max(top, y - 8), value.toFixed(3), { 'text-anchor': 'middle', 'font-size': 10 });
    svgText(token, x + 24, bottom + 18, `t${index + 1}`, { 'text-anchor': 'middle', fill: colors.ink });
    if (index === sampled) svgText(token, x + 24, bottom + 34, 'sampled', { 'text-anchor': 'middle', 'font-size': 9, fill: colors.teal });
    function select() { sampled = index; drawDistribution(); const bars = $$('[role="button"]', $('#distribution-chart')); bars[index].focus({ preventScroll: true }); }
    token.addEventListener('click', select);
    token.addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); select(); } });
    chart.append(token);
  });
  $('#distribution-chart').replaceChildren(chart);
  const name = `${advantage > 0 ? 'Positive' : 'Negative'} ${p[sampled] >= boundary ? 'peak' : 'valley'}`;
  const lead = document.createElement('strong'); lead.textContent = name;
  const direction = Math.abs(hChange) < 1e-10 ? 'is unchanged' : hChange > 0 ? 'increases' : 'decreases';
  const rest = document.createTextNode(`: entropy ${direction} (${hChange >= 0 ? '+' : ''}${hChange.toFixed(4)} nats). ${sampled === 5 ? 'The tiny token is sampled here, so the unsampled-token rule does not apply to it.' : `The tiny unsampled token t6 ${tinyChange > 0 ? 'gains' : 'loses'} ${Math.abs(tinyChange).toExponential(2)} probability mass.`}`);
  $('#lab-result').replaceChildren(lead, rest);
}
$('#distribution').addEventListener('change', () => { sampled = 0; drawDistribution(); });
$('#advantage').addEventListener('change', drawDistribution);
drawDistribution();

// Anchors inside a closed evidence panel should reveal their destination.
function revealHash() {
  if (!location.hash) return;
  let destination;
  try { destination = document.getElementById(decodeURIComponent(location.hash.slice(1))); } catch { return; }
  if (!destination) return;
  for (let ancestor = destination.parentElement; ancestor; ancestor = ancestor.parentElement) if (ancestor.tagName === 'DETAILS') ancestor.open = true;
}
addEventListener('hashchange', revealHash);
revealHash();
