const { test, expect } = require('@playwright/test');
const path = require('path');
const http = require('http');
const fs = require('fs');

let server;
let port;

test.beforeAll(async () => {
  const htmlPath = path.join(__dirname, 'inscription.ultra.html');

  server = http.createServer((req, res) => {
    res.writeHead(200, { 'Content-Type': 'text/html' });
    fs.createReadStream(htmlPath).pipe(res);
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  port = server.address().port;
});

test.afterAll(async () => {
  if (server) server.close();
});

test('ultra inline inscription loads embedded payload and generates text', async ({ page }) => {
  test.setTimeout(120000);

  const fetched = [];
  page.on('request', request => fetched.push(request.url()));

  await page.goto(`http://127.0.0.1:${port}/`);
  await page.waitForLoadState('networkidle');
  await expect(page.locator('#o')).toContainText('Ready.', { timeout: 60000 });
  await expect(page.locator('#g')).toBeEnabled();

  expect(fetched.some(url => url.endsWith('/pleb.slop'))).toBe(false);
  expect(fetched.some(url => url.includes('/content/'))).toBe(false);

  await page.locator('#t').fill('0.75');
  await page.locator('#q').fill('0.85');
  await page.locator('#k').fill('80');
  await page.locator('#f').fill('0.25');
  await page.locator('#g').click();
  await expect(page.locator('#o')).toContainText('bitcoin is', { timeout: 60000 });
  const output = await page.locator('#o').textContent();
  expect(output.length).toBeGreaterThan(30);
});
