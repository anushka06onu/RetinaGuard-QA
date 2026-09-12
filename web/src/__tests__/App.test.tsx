import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import App from '../App';

describe('RetinaGuard-QA Web Frontend', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    globalThis.URL.createObjectURL = vi.fn(() => 'blob:mock-url');
    globalThis.URL.revokeObjectURL = vi.fn();
  });

  it('renders research prototype title and header elements', () => {
    render(<App />);
    expect(screen.getAllByText(/RetinaGuard/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/Fundus Image QA \(Research Prototype\)/i)).toBeInTheDocument();
    expect(screen.getByText(/Three Pillars of Technical Reliability/i)).toBeInTheDocument();
  });

  it('validates file types and rejects non-jpeg/png files', () => {
    render(<App />);
    const fileInput = screen.getByTestId('file-input');

    const invalidFile = new File(['text content'], 'document.pdf', { type: 'application/pdf' });
    fireEvent.change(fileInput, { target: { files: [invalidFile] } });

    expect(screen.getByText(/Please select a valid image file/i)).toBeInTheDocument();
  });

  it('rejects files exceeding 15 MB limit', () => {
    render(<App />);
    const fileInput = screen.getByTestId('file-input');

    // Create a 16 MB mock file
    const largeFile = new File([new ArrayBuffer(16 * 1024 * 1024)], 'large_fundus.jpg', { type: 'image/jpeg' });
    fireEvent.change(fileInput, { target: { files: [largeFile] } });

    expect(screen.getByText(/File is too large/i)).toBeInTheDocument();
    expect(screen.getByText(/Maximum allowed upload size is 15 MB/i)).toBeInTheDocument();
  });

  it('accepts valid JPEG file and updates preview and clears with reset button', () => {
    render(<App />);
    const fileInput = screen.getByTestId('file-input');

    const validFile = new File(['fake-image-bytes'], 'fundus.jpg', { type: 'image/jpeg' });
    fireEvent.change(fileInput, { target: { files: [validFile] } });

    expect(screen.queryByText(/Please select a valid image file/i)).not.toBeInTheDocument();
    expect(screen.getByAltText('Preview')).toBeInTheDocument();

    const clearBtn = screen.getByText(/Clear/i);
    fireEvent.click(clearBtn);
    expect(screen.queryByAltText('Preview')).not.toBeInTheDocument();
  });

  it('supports accessible keyboard trigger on dropzone', () => {
    render(<App />);
    const dropzone = screen.getByTestId('dropzone');
    const fileInput = screen.getByTestId('file-input');
    const clickSpy = vi.spyOn(fileInput, 'click');

    fireEvent.keyDown(dropzone, { key: 'Enter', code: 'Enter' });
    expect(clickSpy).toHaveBeenCalled();
  });

  it('handles successful API prediction response and renders decision badges', async () => {
    const mockPrediction = {
      model_version: '0.2.0',
      quality: 'good',
      probabilities: { good: 0.92, usable: 0.06, reject: 0.02 },
      calibrated_confidence: 0.92,
      uncertainty: 0.142,
      ood_score: -12.45,
      decision: 'accept',
      quality_attributes: {
        artifact: 'none',
        clarity: 'high',
        field_definition: 'adequate'
      },
      feedback: ['Optimal macular center', 'Sharp vascular contrast'],
      disclaimer: 'Technical image-quality assessment only; not a clinical diagnosis or treatment recommendation.',
      latency_ms: 18.5
    };

    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockPrediction,
    } as Response);

    render(<App />);
    const fileInput = screen.getByTestId('file-input');
    const validFile = new File(['fake-image-bytes'], 'fundus.jpg', { type: 'image/jpeg' });
    fireEvent.change(fileInput, { target: { files: [validFile] } });

    const predictBtn = screen.getByTestId('predict-button');
    fireEvent.click(predictBtn);

    await waitFor(() => {
      expect(screen.getByTestId('decision-badge')).toBeInTheDocument();
      expect(screen.getByText(/Decision: Accept/i)).toBeInTheDocument();
      expect(screen.getByText(/92.0% Conf/i)).toBeInTheDocument();
      expect(screen.getByText(/0.142/i)).toBeInTheDocument();
      expect(screen.getByText(/Optimal macular center/i)).toBeInTheDocument();
      expect(screen.getByText(/Model: v0.2.0/i)).toBeInTheDocument();
    });
  });

  it('handles backend 503 readiness error gracefully', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 503,
      json: async () => ({ status: 'not_ready' }),
    } as Response);

    render(<App />);
    const fileInput = screen.getByTestId('file-input');
    const validFile = new File(['bytes'], 'fundus.png', { type: 'image/png' });
    fireEvent.change(fileInput, { target: { files: [validFile] } });

    fireEvent.click(screen.getByTestId('predict-button'));

    await waitFor(() => {
      expect(screen.getByText(/Backend service is not ready/i)).toBeInTheDocument();
    });
  });

  it('renders recapture badge correctly on recapture decision', async () => {
    const mockRecapture = {
      model_version: '0.2.0',
      quality: 'reject',
      probabilities: { good: 0.05, usable: 0.15, reject: 0.80 },
      calibrated_confidence: 0.80,
      uncertainty: 0.38,
      ood_score: -5.1,
      decision: 'recapture',
      quality_attributes: {
        artifact: 'severe',
        clarity: 'low',
        field_definition: 'poor'
      },
      feedback: ['Severe defocus detected. Clean objective lens and refocus.'],
      disclaimer: 'Technical image-quality assessment only.',
      latency_ms: 19.2
    };

    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockRecapture,
    } as Response);

    render(<App />);
    const fileInput = screen.getByTestId('file-input');
    const validFile = new File(['bytes'], 'blur.jpg', { type: 'image/jpeg' });
    fireEvent.change(fileInput, { target: { files: [validFile] } });

    fireEvent.click(screen.getByTestId('predict-button'));

    await waitFor(() => {
      expect(screen.getByText(/Decision: Recapture Recommended/i)).toBeInTheDocument();
      expect(screen.getByText(/Severe defocus detected/i)).toBeInTheDocument();
    });
  });
});

