const { test, expect } = require('@playwright/test');
const path = require('path');
const http = require('http');
const fs = require('fs');

let server;
let port;

test.beforeAll(async () => {
  const htmlPath = path.join(__dirname, 'inscription.inline.html');

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

test('inline inscription loads embedded model without fetching weights', async ({ page }) => {
  const fetched = [];
  page.on('request', request => fetched.push(request.url()));

  await page.goto(`http://127.0.0.1:${port}/`);
  await page.waitForLoadState('networkidle');

  await page.waitForFunction(
    () => document.querySelector('#status').classList.contains('ready'),
    {},
    { timeout: 60000 }
  );

  await expect(page.locator('#status')).toHaveClass(/ready/);
  await expect(page.locator('#generateBtn')).toBeEnabled();
  expect(fetched.some(url => url.endsWith('/pleb.slop'))).toBe(false);
  expect(fetched.some(url => url.includes('/content/'))).toBe(false);
});
