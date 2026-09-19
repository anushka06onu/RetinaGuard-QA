import { test, expect } from '@playwright/test';
import * as fs from 'fs';
import * as path from 'path';

test.describe('RetinaGuard-QA Full-Stack End-to-End Test', () => {
  test('verifies backend readiness, uploads fundus fixture, and inspects quality', async ({ page }) => {
    // 1. Navigate to the frontend application
    await page.goto('/');

    // 2. Verify page titles and hero structure
    await expect(page.locator('h1')).toContainText('Uncertainty-Aware Quality Control');
    await expect(page.getByText('Three Pillars of Technical Reliability')).toBeVisible();

    // 3. Verify Backend Readiness indicator shows Model Ready
    await expect(page.getByText('Model Ready')).toBeVisible({ timeout: 15000 });

    // 4. Create a 100x100 synthetic fundus-like image buffer (>32x32 minimum dimension)
    const testImageBase64 = await page.evaluate(() => {
      const canvas = document.createElement('canvas');
      canvas.width = 100;
      canvas.height = 100;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.fillStyle = '#111827';
        ctx.fillRect(0, 0, 100, 100);
        ctx.fillStyle = '#b43214';
        ctx.beginPath();
        ctx.arc(50, 50, 40, 0, Math.PI * 2);
        ctx.fill();
      }
      return canvas.toDataURL('image/png').split(',')[1];
    });
    const testImageBuffer = Buffer.from(testImageBase64, 'base64');

    // 5. Upload image using file chooser / input
    const fileInput = page.locator('input[data-testid="file-input"]');
    await fileInput.setInputFiles({
      name: 'sample_fundus.png',
      mimeType: 'image/png',
      buffer: testImageBuffer,
    });

    // 6. Preview should be visible
    await expect(page.locator('img[alt*="sample_fundus.png"]')).toBeVisible();

    // 7. Predict button should now be enabled
    const predictBtn = page.locator('button[data-testid="predict-button"]');
    await expect(predictBtn).toBeEnabled();

    // 8. Trigger inference inspection
    await predictBtn.click();

    // 9. Verify decision badge and prediction results
    const decisionBadge = page.locator('div[data-testid="decision-badge"]');
    await expect(decisionBadge).toBeVisible({ timeout: 15000 });

    // Verify key UI result cards
    await expect(page.getByText('Quality Grade', { exact: true })).toBeVisible();
    await expect(page.getByText('Uncertainty', { exact: true })).toBeVisible();
    await expect(page.getByText('Inference Latency', { exact: true })).toBeVisible();
    await expect(page.getByText('Class Probabilities', { exact: true })).toBeVisible();
    await expect(page.getByText('Actionable Capture Guidance', { exact: true })).toBeVisible();
    await expect(page.getByText('Research Log:', { exact: false })).toBeVisible();

    // Verify non-diagnostic disclaimer
    await expect(page.getByText(/Technical image-quality assessment only/i)).toBeVisible();

    // 10. Test invalid file selection clears results and shows error
    await fileInput.setInputFiles({
      name: 'invalid_doc.pdf',
      mimeType: 'application/pdf',
      buffer: Buffer.from('invalid file contents'),
    });

    await expect(page.getByText('Please select a valid image file')).toBeVisible();
    await expect(decisionBadge).not.toBeVisible();
  });
});
