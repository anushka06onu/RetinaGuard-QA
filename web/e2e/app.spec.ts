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

    // 4. Create a mock synthetic fundus JPEG image buffer
    const testImageBuffer = Buffer.from(
      '/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////wgALCAABAAEBAREA/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPxA=',
      'base64'
    );

    // 5. Upload image using file chooser / input
    const fileInput = page.locator('input[data-testid="file-input"]');
    await fileInput.setInputFiles({
      name: 'sample_fundus.jpg',
      mimeType: 'image/jpeg',
      buffer: testImageBuffer,
    });

    // 6. Preview should be visible
    await expect(page.locator('img[alt*="sample_fundus.jpg"]')).toBeVisible();

    // 7. Predict button should now be enabled
    const predictBtn = page.locator('button[data-testid="predict-button"]');
    await expect(predictBtn).toBeEnabled();

    // 8. Trigger inference inspection
    await predictBtn.click();

    // 9. Verify decision badge and prediction results
    const decisionBadge = page.locator('div[data-testid="decision-badge"]');
    await expect(decisionBadge).toBeVisible({ timeout: 15000 });

    // Verify key UI result cards
    await expect(page.getByText('Quality Grade')).toBeVisible();
    await expect(page.getByText('Uncertainty')).toBeVisible();
    await expect(page.getByText('Inference Latency')).toBeVisible();
    await expect(page.getByText('Class Probabilities')).toBeVisible();
    await expect(page.getByText('Actionable Capture Guidance')).toBeVisible();
    await expect(page.getByText('Research Log:')).toBeVisible();

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
