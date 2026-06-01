const { test, expect } = require('@playwright/test');
const path = require('path');

test('inscription.html smoke test', async ({ page }) => {
  const htmlPath = path.join(__dirname, 'inscription.html');
  await page.goto(`file://${htmlPath}`);
  
  // Wait for page to load
  await page.waitForLoadState('networkidle');
  await page.waitForTimeout(1000);
  
  // Assert the container is visible and has non-zero dimensions
  const container = page.locator('.container');
  await expect(container).toBeVisible();
  const containerBox = await container.boundingBox();
  expect(containerBox).not.toBeNull();
  expect(containerBox.width).toBeGreaterThan(0);
  expect(containerBox.height).toBeGreaterThan(0);

  // Assert the h1 heading is visible
  const h1 = page.locator('h1');
  await expect(h1).toBeVisible();
  const h1Box = await h1.boundingBox();
  expect(h1Box).not.toBeNull();
  expect(h1Box.width).toBeGreaterThan(0);
  expect(h1Box.height).toBeGreaterThan(0);

  // Assert the generate button exists and is visible
  const generateBtn = page.locator('#generateBtn');
  await expect(generateBtn).toBeVisible();
  const btnBox = await generateBtn.boundingBox();
  expect(btnBox).not.toBeNull();
  expect(btnBox.width).toBeGreaterThan(0);
  expect(btnBox.height).toBeGreaterThan(0);

  // Assert the page body is NOT just a solid color
  // Check that there are visible text elements in the viewport
  const bodyText = await page.locator('body').innerText();
  expect(bodyText).toBeTruthy();
  expect(bodyText.length).toBeGreaterThan(10);

  // Additional check: verify the page has actual content (not blank/orange/broken)
  // The page should have the PlebGPT title
  const titleText = await h1.textContent();
  expect(titleText).toContain('PlebGPT');
});

// E2E test with ord regtest server
test('e2e regtest test - inscribe UI and weights, verify in browser', async ({ page }) => {
  // Skip if not running against regtest server
  test.skip(!process.env.REGTEST_URL, 'REGTEST_URL not set - skipping e2e test');
  
  const baseUrl = process.env.REGTEST_URL || 'http://localhost:9001';
  
  // Get inscription IDs from environment
  const parentId = process.env.PARENT_INSCRIPTION_ID || '';
  const childId = process.env.CHILD_INSCRIPTION_ID || '';
  
  test.skip(!parentId || !childId, 'Inscription IDs not set - skipping e2e test');
  
  // Navigate to the inscribed UI (parent)
  await page.goto(`${baseUrl}/content/${parentId}`);
  
  // Wait for page to load
  await page.waitForLoadState('networkidle');
  await page.waitForTimeout(2000);
  
  // Verify the UI loads - container should be visible
  const container = page.locator('.container');
  await expect(container).toBeVisible();
  const containerBox = await container.boundingBox();
  expect(containerBox).not.toBeNull();
  expect(containerBox.width).toBeGreaterThan(0);
  expect(containerBox.height).toBeGreaterThan(0);
  
  // Verify the heading is visible
  const h1 = page.locator('h1');
  await expect(h1).toBeVisible();
  const h1Box = await h1.boundingBox();
  expect(h1Box).not.toBeNull();
  expect(h1Box.width).toBeGreaterThan(0);
  expect(h1Box.height).toBeGreaterThan(0);
  
  const titleText = await h1.textContent();
  expect(titleText).toContain('PlebGPT');
  
  // Wait for model to load - status should change to 'ready'
  // The model loads from the child inscription
  const statusEl = page.locator('.status');
  
  // Wait for loading status first
  await expect(statusEl).toBeVisible();
  
  // Wait for model to be ready (up to 30 seconds)
  await page.waitForFunction(
    () => {
      const status = document.querySelector('.status');
      return status && status.classList.contains('ready');
    },
    {},
    { timeout: 30000 }
  );
  
  // Verify status shows ready
  await expect(statusEl).toHaveClass(/ready/);
  const statusText = await statusEl.textContent();
  expect(statusText).toContain('ready');
  
  // Verify generate button is enabled
  const generateBtn = page.locator('#generateBtn');
  await expect(generateBtn).toBeEnabled();
  
  // Test generation with the default prompt.
  await page.locator('#promptSelect').selectOption('bitcoin is');
  
  // Click generate button
  await generateBtn.click();
  
  // Wait for output to appear
  const outputEl = page.locator('.output');
  await expect(outputEl).toBeVisible();
  
  // Wait for non-empty output (up to 30 seconds for generation)
  await page.waitForFunction(
    () => {
      const output = document.querySelector('.output');
      return output && output.textContent.trim().length > 10;
    },
    {},
    { timeout: 30000 }
  );
  
  // Verify output is non-empty and contains generated text
  const outputText = await outputEl.textContent();
  expect(outputText.length).toBeGreaterThan(10);
  expect(outputText).toContain('bitcoin'); // Should echo the prompt
  
  // Verify the page is not broken (not all orange, not blank)
  const bodyText = await page.locator('body').innerText();
  expect(bodyText.length).toBeGreaterThan(50);
  
  // Verify we can see actual content (not a solid color page)
  const h1Visible = await h1.isVisible();
  const containerVisible = await container.isVisible();
  expect(h1Visible).toBe(true);
  expect(containerVisible).toBe(true);
});

// Test the children endpoint
test('children endpoint returns weights inscription', async ({ request }) => {
  // Skip if not running against regtest server
  test.skip(!process.env.REGTEST_URL, 'REGTEST_URL not set - skipping endpoint test');
  
  const parentId = process.env.PARENT_INSCRIPTION_ID || '';
  test.skip(!parentId, 'PARENT_INSCRIPTION_ID not set - skipping endpoint test');
  
  const baseUrl = process.env.REGTEST_URL || 'http://localhost:9001';
  
  // Call the children endpoint
  const response = await request.get(`${baseUrl}/r/children/${parentId}`);
  expect(response.ok()).toBeTruthy();
  
  const data = await response.json();
  
  // The response should contain the child inscription (weights)
  // This verifies the parent-child relationship between UI and model inscriptions
  expect(data).toBeDefined();
  expect(data.ids).toBeDefined();
  expect(Array.isArray(data.ids)).toBe(true);
  
  // Should have at least one child (the model weights)
  expect(data.ids.length).toBeGreaterThan(0);
  
  // Verify the child ID is in the response
  const childId = process.env.CHILD_INSCRIPTION_ID || '';
  if (childId) {
    expect(data.ids).toContain(childId);

    const childResponse = await request.get(`${baseUrl}/r/inscription/${childId}`);
    expect(childResponse.ok()).toBeTruthy();

    const childMetadata = await childResponse.json();
    expect(childMetadata.content_type).toBe('application/octet-stream');
  }
});
