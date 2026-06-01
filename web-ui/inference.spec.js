const { test, expect } = require('@playwright/test');
const path = require('path');
const http = require('http');
const fs = require('fs');

let server;
let port;

test.beforeAll(async () => {
  const slopPath = path.join(__dirname, '..', 'pleb.slop');
  const htmlPath = path.join(__dirname, 'inscription.html');

  server = http.createServer((req, res) => {
    if (req.url === '/pleb.slop') {
      res.writeHead(200, { 'Content-Type': 'application/octet-stream' });
      fs.createReadStream(slopPath).pipe(res);
    } else {
      res.writeHead(200, { 'Content-Type': 'text/html' });
      fs.createReadStream(htmlPath).pipe(res);
    }
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  port = server.address().port;
});

test.afterAll(async () => {
  if (server) server.close();
});

test('JS inference produces readable output', async ({ page }) => {
  test.setTimeout(300000);

  // Capture console and errors
  const consoleLogs = [];
  const errors = [];
  page.on('console', msg => consoleLogs.push(`${msg.type()}: ${msg.text()}`));
  page.on('pageerror', err => errors.push(err.message));

  await page.goto(`http://127.0.0.1:${port}/`);
  await page.waitForLoadState('networkidle');

  // Wait for model ready
  await page.waitForFunction(
    () => document.querySelector('#status').classList.contains('ready'),
    {},
    { timeout: 60000 }
  );
  console.log('Model loaded and ready');

  // Select custom prompt and fill
  await page.locator('#promptSelect').selectOption('custom');
  await page.locator('#customPrompt').fill('bitcoin is');

  // Click generate
  await page.locator('#generateBtn').click();
  console.log('Generate button clicked');

  // Monitor progress — check output periodically
  let outputText = '';
  for (let i = 0; i < 150; i++) {
    await page.waitForTimeout(2000);
    outputText = await page.locator('#output').textContent();
    console.log(`  [${i * 2}s] output length: ${outputText.length}, text: ${outputText.substring(0, 80)}`);
    if (outputText.includes('Time:')) break;
    if (outputText.includes('Error:')) {
      console.log('Generation error detected:', outputText);
      break;
    }
  }

  // Report any JS errors
  if (errors.length) {
    console.log('JS errors:', errors);
  }

  // Show last few console logs
  console.log('Last console logs:', consoleLogs.slice(-10));

  console.log('Final output:', outputText);

  // Verify basic generation worked
  expect(outputText).toBeTruthy();
  expect(outputText).toContain('bitcoin is');
  expect(outputText.length).toBeGreaterThan(30);

  // Check for English words (quality gate)
  const englishWords = ['and', 'the', 'is', 'are', 'of', 'in', 'that', 'to', 'it', 'have', 'this', 'not', 'for', 'with'];
  const foundWords = englishWords.filter(w => outputText.toLowerCase().includes(w));
  console.log('Found English words:', foundWords);
  expect(foundWords.length).toBeGreaterThanOrEqual(5);

  // Check jargon ratio
  const jargonWords = ['thermodynamic', 'cyberhornets', 'nocoiners', 'uninflatable', 'uncorruptible', 'cantillon', 'hyperbitcoinization'];
  const outputLower = outputText.toLowerCase();
  let jargonCount = 0;
  jargonWords.forEach(w => {
    const matches = outputLower.match(new RegExp(w, 'g'));
    if (matches) jargonCount += matches.length;
  });
  const allWords = outputLower.match(/\b\w+\b/g) || [];
  const generatedWords = allWords.slice(2);
  const jargonRatio = jargonCount / Math.max(generatedWords.length, 1);
  console.log(`Jargon ratio: ${(jargonRatio * 100).toFixed(1)}%`);
  expect(jargonRatio).toBeLessThan(0.5);

  // No 3-word loops
  const words = generatedWords.map(w => w.replace(/[^a-z]/g, ''));
  let hasLoop = false;
  for (let i = 0; i < words.length - 2; i++) {
    if (words[i] && words[i] === words[i + 1] && words[i] === words[i + 2]) {
      hasLoop = true;
      console.log(`Found loop: "${words[i]}"`);
    }
  }
  expect(hasLoop).toBe(false);
});

test('max tokens control limits generated output', async ({ page }) => {
  test.setTimeout(120000);

  await page.goto(`http://127.0.0.1:${port}/`);
  await page.waitForLoadState('networkidle');

  await page.waitForFunction(
    () => document.querySelector('#status').classList.contains('ready'),
    {},
    { timeout: 60000 }
  );

  await page.locator('#advancedPanel').evaluate(el => el.open = true);
  await page.locator('#maxTokensInput').fill('1');
  await page.locator('#promptSelect').selectOption('custom');
  await page.locator('#customPrompt').fill('bitcoin is');
  await page.locator('#generateBtn').click();

  await page.waitForFunction(
    () => document.querySelector('#output').textContent.includes('Time:'),
    {},
    { timeout: 60000 }
  );

  const outputText = await page.locator('#output').textContent();
  const generatedLine = outputText.split('\n\n(Time:')[0];
  const generatedText = generatedLine.substring('bitcoin is'.length).trim();

  expect(outputText).toContain('bitcoin is');
  expect(generatedText.length).toBeGreaterThan(0);
  expect(generatedText.length).toBeLessThan(40);
});
