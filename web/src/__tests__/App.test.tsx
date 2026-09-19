import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import App, { validatePredictionResponse } from '../App';

describe('RetinaGuard-QA Web Frontend', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    globalThis.URL.createObjectURL = vi.fn(() => 'blob:mock-url');
    globalThis.URL.revokeObjectURL = vi.fn();
    // Default mock for /health/ready check returning genuine ready payload
    globalThis.fetch = vi.fn().mockImplementation((url: string) => {
      if (typeof url === 'string' && url.includes('/health/ready')) {
        return Promise.resolve({
          ok: true,
          status: 200,
          json: async () => ({ status: 'ready', model_loaded: true }),
        } as Response);
      }
      return Promise.resolve({
        ok: true,
        status: 200,
        json: async () => ({}),
      } as Response);
    });
  });

  it('renders research prototype title and accurate benchmark figures', async () => {
    render(<App />);
    await waitFor(() => {
      expect(screen.getByText(/Model Ready/i)).toBeInTheDocument();
    });
    expect(screen.getAllByText(/RetinaGuard/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/Fundus Image QA \(Research Prototype\)/i)).toBeInTheDocument();
    expect(screen.getByText(/Three Pillars of Technical Reliability/i)).toBeInTheDocument();
    expect(screen.getByText(/8\.42 ms and throughput of 20\.14 images\/sec/i)).toBeInTheDocument();
  });

  it('validates file types and rejects non-jpeg/png files', async () => {
    render(<App />);
    await waitFor(() => {
      expect(screen.getByText(/Model Ready/i)).toBeInTheDocument();
    });
    const fileInput = screen.getByTestId('file-input');

    const invalidFile = new File(['text content'], 'document.pdf', { type: 'application/pdf' });
    fireEvent.change(fileInput, { target: { files: [invalidFile] } });

    expect(screen.getByText(/Please select a valid image file/i)).toBeInTheDocument();
  });

  it('rejects files exceeding 15 MB limit', async () => {
    render(<App />);
    await waitFor(() => {
      expect(screen.getByText(/Model Ready/i)).toBeInTheDocument();
    });
    const fileInput = screen.getByTestId('file-input');

    // Create a 16 MB mock file
    const largeFile = new File([new ArrayBuffer(16 * 1024 * 1024)], 'large_fundus.jpg', { type: 'image/jpeg' });
    fireEvent.change(fileInput, { target: { files: [largeFile] } });

    expect(screen.getByText(/File is too large/i)).toBeInTheDocument();
    expect(screen.getByText(/Maximum allowed upload size is 15 MB/i)).toBeInTheDocument();
  });

  it('accepts valid JPEG file and updates preview and clears with reset button', async () => {
    render(<App />);
    await waitFor(() => {
      expect(screen.getByText(/Model Ready/i)).toBeInTheDocument();
    });
    const fileInput = screen.getByTestId('file-input');

    const validFile = new File(['fake-image-bytes'], 'fundus.jpg', { type: 'image/jpeg' });
    fireEvent.change(fileInput, { target: { files: [validFile] } });

    expect(screen.queryByText(/Please select a valid image file/i)).not.toBeInTheDocument();
    expect(screen.getByAltText(/Preview of selected fundus image: fundus\.jpg/i)).toBeInTheDocument();

    const clearBtn = screen.getByText(/Clear/i);
    fireEvent.click(clearBtn);
    expect(screen.queryByAltText(/Preview of selected fundus image/i)).not.toBeInTheDocument();
  });

  it('clears previous successful prediction result when user selects an invalid file', async () => {
    const mockPrediction = {
      model_version: '1.0.2',
      quality: 'good',
      probabilities: { good: 0.95, poor_or_reject: 0.05 },
      calibrated_confidence: 0.95,
      uncertainty: 0.12,
      ood_score: -10.5,
      decision: 'accept',
      quality_attributes: {
        artifact: 'none',
        clarity: 'high',
        field_definition: 'adequate'
      },
      feedback: ['Optimal macular center'],
      disclaimer: 'Technical image-quality assessment only.',
      latency_ms: 8.42
    };

    globalThis.fetch = vi.fn().mockImplementation((url: string) => {
      if (typeof url === 'string' && url.includes('/health/ready')) {
        return Promise.resolve({ ok: true, json: async () => ({ status: 'ready' }) } as Response);
      }
      return Promise.resolve({ ok: true, json: async () => mockPrediction } as Response);
    });

    render(<App />);
    await waitFor(() => {
      expect(screen.getByText(/Model Ready/i)).toBeInTheDocument();
    });
    const fileInput = screen.getByTestId('file-input');
    const validFile = new File(['bytes'], 'fundus.jpg', { type: 'image/jpeg' });
    fireEvent.change(fileInput, { target: { files: [validFile] } });

    fireEvent.click(screen.getByTestId('predict-button'));

    await waitFor(() => {
      expect(screen.getByTestId('decision-badge')).toBeInTheDocument();
    });

    // User now selects an invalid PDF file
    const invalidFile = new File(['pdf-bytes'], 'invalid.pdf', { type: 'application/pdf' });
    fireEvent.change(fileInput, { target: { files: [invalidFile] } });

    // Result should be cleared immediately
    expect(screen.queryByTestId('decision-badge')).not.toBeInTheDocument();
    expect(screen.getByText(/Please select a valid image file/i)).toBeInTheDocument();
  });

  it('clears previous successful prediction result when user selects an oversized file', async () => {
    const mockPrediction = {
      model_version: '1.0.2',
      quality: 'good',
      probabilities: { good: 0.95, poor_or_reject: 0.05 },
      calibrated_confidence: 0.95,
      uncertainty: 0.12,
      ood_score: -10.5,
      decision: 'accept',
      quality_attributes: {
        artifact: 'none',
        clarity: 'high',
        field_definition: 'adequate'
      },
      feedback: ['Optimal macular center'],
      disclaimer: 'Technical image-quality assessment only.',
      latency_ms: 8.42
    };

    globalThis.fetch = vi.fn().mockImplementation((url: string) => {
      if (typeof url === 'string' && url.includes('/health/ready')) {
        return Promise.resolve({ ok: true, json: async () => ({ status: 'ready' }) } as Response);
      }
      return Promise.resolve({ ok: true, json: async () => mockPrediction } as Response);
    });

    render(<App />);
    await waitFor(() => {
      expect(screen.getByText(/Model Ready/i)).toBeInTheDocument();
    });
    const fileInput = screen.getByTestId('file-input');
    const validFile = new File(['bytes'], 'fundus.jpg', { type: 'image/jpeg' });
    fireEvent.change(fileInput, { target: { files: [validFile] } });

    fireEvent.click(screen.getByTestId('predict-button'));

    await waitFor(() => {
      expect(screen.getByTestId('decision-badge')).toBeInTheDocument();
    });

    // User now selects an oversized file
    const largeFile = new File([new ArrayBuffer(16 * 1024 * 1024)], 'large.jpg', { type: 'image/jpeg' });
    fireEvent.change(fileInput, { target: { files: [largeFile] } });

    expect(screen.queryByTestId('decision-badge')).not.toBeInTheDocument();
    expect(screen.getByText(/File is too large/i)).toBeInTheDocument();
  });

  it('supports accessible keyboard trigger on dropzone', async () => {
    render(<App />);
    await waitFor(() => {
      expect(screen.getByText(/Model Ready/i)).toBeInTheDocument();
    });
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

    globalThis.fetch = vi.fn().mockImplementation((url: string) => {
      if (typeof url === 'string' && url.includes('/health/ready')) {
        return Promise.resolve({ ok: true, json: async () => ({ status: 'ready' }) } as Response);
      }
      return Promise.resolve({ ok: true, json: async () => mockPrediction } as Response);
    });

    render(<App />);
    await waitFor(() => {
      expect(screen.getByText(/Model Ready/i)).toBeInTheDocument();
    });
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
      expect(screen.getByText(/18.5/i)).toBeInTheDocument();
    });
  });

  it('handles voluntary cancellation cleanly without showing error banner', async () => {
    let abortSignal: AbortSignal | undefined;
    globalThis.fetch = vi.fn().mockImplementation((url: string, init?: RequestInit) => {
      if (typeof url === 'string' && url.includes('/health/ready')) {
        return Promise.resolve({ ok: true, json: async () => ({ status: 'ready' }) } as Response);
      }
      abortSignal = init?.signal as AbortSignal;
      return new Promise((_, reject) => {
        if (abortSignal) {
          abortSignal.addEventListener('abort', () => {
            const err = new Error('The operation was aborted');
            err.name = 'AbortError';
            reject(err);
          });
        }
      });
    });

    render(<App />);
    await waitFor(() => {
      expect(screen.getByText(/Model Ready/i)).toBeInTheDocument();
    });
    const fileInput = screen.getByTestId('file-input');
    const validFile = new File(['bytes'], 'fundus.jpg', { type: 'image/jpeg' });
    fireEvent.change(fileInput, { target: { files: [validFile] } });

    fireEvent.click(screen.getByTestId('predict-button'));

    const cancelBtn = await screen.findByText(/Cancel/i);
    fireEvent.click(cancelBtn);

    await waitFor(() => {
      expect(screen.getByText(/Inspection cancelled/i)).toBeInTheDocument();
      expect(screen.queryByText(/Inference request timed out/i)).not.toBeInTheDocument();
    });
  });

  it('handles binary DeepDRiD prediction response and renders quality and attributes', async () => {
    const mockBinaryPrediction = {
      model_version: '0.2.0',
      quality: 'poor_or_reject',
      probabilities: { good: 0.12, poor_or_reject: 0.88 },
      calibrated_confidence: 0.88,
      uncertainty: 0.529,
      ood_score: -1.2,
      decision: 'recapture',
      quality_attributes: {
        artifact: 'mild',
        clarity: 'low',
        field_definition: 'adequate'
      },
      feedback: ['Possible blur or focus defect detected. Stabilize and refocus camera before recapture.'],
      disclaimer: 'Technical image-quality assessment only; not a clinical diagnosis or treatment recommendation.',
      latency_ms: 8.2
    };

    globalThis.fetch = vi.fn().mockImplementation((url: string) => {
      if (typeof url === 'string' && url.includes('/health/ready')) {
        return Promise.resolve({ ok: true, json: async () => ({ status: 'ready' }) } as Response);
      }
      return Promise.resolve({ ok: true, json: async () => mockBinaryPrediction } as Response);
    });

    render(<App />);
    await waitFor(() => {
      expect(screen.getByText(/Model Ready/i)).toBeInTheDocument();
    });
    const fileInput = screen.getByTestId('file-input');
    const validFile = new File(['fake-bytes'], 'fundus.jpg', { type: 'image/jpeg' });
    fireEvent.change(fileInput, { target: { files: [validFile] } });

    const predictBtn = screen.getByTestId('predict-button');
    fireEvent.click(predictBtn);

    await waitFor(() => {
      expect(screen.getByTestId('decision-badge')).toBeInTheDocument();
      expect(screen.getAllByText(/Poor \/ Reject/i).length).toBeGreaterThan(0);
      expect(screen.getByText(/88.0% Conf/i)).toBeInTheDocument();
      expect(screen.getByText(/0.529/i)).toBeInTheDocument();
      expect(screen.getByText(/Possible blur or focus defect detected/i)).toBeInTheDocument();
      expect(screen.getByText(/8.2/i)).toBeInTheDocument();
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

    globalThis.fetch = vi.fn().mockImplementation((url: string) => {
      if (typeof url === 'string' && url.includes('/health/ready')) {
        return Promise.resolve({ ok: true, json: async () => ({ status: 'ready' }) } as Response);
      }
      return Promise.resolve({ ok: true, json: async () => mockRecapture } as Response);
    });

    render(<App />);
    await waitFor(() => {
      expect(screen.getByText(/Model Ready/i)).toBeInTheDocument();
    });
    const fileInput = screen.getByTestId('file-input');
    const validFile = new File(['bytes'], 'blur.jpg', { type: 'image/jpeg' });
    fireEvent.change(fileInput, { target: { files: [validFile] } });

    fireEvent.click(screen.getByTestId('predict-button'));

    await waitFor(() => {
      expect(screen.getByText(/Decision: Recapture Recommended/i)).toBeInTheDocument();
      expect(screen.getByText(/Severe defocus detected/i)).toBeInTheDocument();
    });
  });

  it('handles drag and drop file uploads', async () => {
    render(<App />);
    await waitFor(() => {
      expect(screen.getByText(/Model Ready/i)).toBeInTheDocument();
    });
    const dropzone = screen.getByTestId('dropzone');
    const validFile = new File(['dropped-bytes'], 'dropped.png', { type: 'image/png' });

    fireEvent.dragOver(dropzone);
    fireEvent.drop(dropzone, {
      dataTransfer: {
        files: [validFile],
      },
    });

    expect(screen.getByAltText(/Preview of selected fundus image: dropped\.png/i)).toBeInTheDocument();
  });

  it('rejects malformed backend responses with entropy exceeding 1.0 bit for binary predictions', () => {
    const invalidBinaryResponse = {
      model_version: '1.0.2',
      quality: 'good',
      probabilities: { good: 0.5, poor_or_reject: 0.5 },
      calibrated_confidence: 0.5,
      uncertainty: 1.25, // Invalid: exceeds max binary entropy log2(2) = 1.0 bit
      ood_score: -1.0,
      decision: 'accept',
      quality_attributes: {
        artifact: 'none',
        clarity: 'high',
        field_definition: 'adequate'
      },
      feedback: [],
      disclaimer: 'test',
    };

    expect(() => validatePredictionResponse(invalidBinaryResponse)).toThrow(
      /uncertainty \(1\.2500\) must satisfy 0 <= uncertainty <= log2\(2\)/i
    );
  });

  it('strictly validates numeric types and rejects string numbers', () => {
    const stringLatencyResponse = {
      model_version: '1.0.2',
      quality: 'good',
      probabilities: { good: 0.9, poor_or_reject: 0.1 },
      calibrated_confidence: 0.9,
      uncertainty: 0.3,
      ood_score: -1.0,
      decision: 'accept',
      quality_attributes: {
        artifact: 'none',
        clarity: 'high',
        field_definition: 'adequate'
      },
      feedback: [],
      disclaimer: 'test',
      latency_ms: "8.42" // String instead of number
    };

    expect(() => validatePredictionResponse(stringLatencyResponse)).toThrow(
      /latency_ms must be a non-negative finite number/i
    );

    const stringConfidenceResponse = {
      ...stringLatencyResponse,
      latency_ms: 8.42,
      calibrated_confidence: "0.9" // String instead of number
    };

    expect(() => validatePredictionResponse(stringConfidenceResponse)).toThrow(
      /calibrated_confidence must be a finite number between 0 and 1/i
    );

    const stringProbabilityResponse = {
      ...stringLatencyResponse,
      latency_ms: 8.42,
      probabilities: { good: "0.9", poor_or_reject: 0.1 }
    };

    expect(() => validatePredictionResponse(stringProbabilityResponse)).toThrow(
      /good probability must be a finite number between 0 and 1/i
    );
  });

  it('renders and toggles developer/test fixtures accordion', async () => {
    render(<App />);
    await waitFor(() => {
      expect(screen.getByText(/Model Ready/i)).toBeInTheDocument();
    });
    expect(screen.getByText(/Developer \/ Test Fixtures/i)).toBeInTheDocument();
    expect(screen.getByText(/Synthetic canvas patterns for API\/UI verification only/i)).toBeInTheDocument();
    expect(screen.getByTestId('fixture-good')).toBeInTheDocument();
    expect(screen.getByTestId('fixture-blur')).toBeInTheDocument();
    expect(screen.getByTestId('fixture-underexposed')).toBeInTheDocument();
    expect(screen.getByTestId('fixture-ood')).toBeInTheDocument();
  });

  it('disables prediction button when backend is not ready or offline', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ status: 'not_ready' }),
    } as Response);

    render(<App />);
    await waitFor(() => {
      expect(screen.getByText(/Initializing\.\.\./i)).toBeInTheDocument();
    });

    const fileInput = screen.getByTestId('file-input');
    const validFile = new File(['bytes'], 'fundus.jpg', { type: 'image/jpeg' });
    fireEvent.change(fileInput, { target: { files: [validFile] } });

    const predictBtn = screen.getByTestId('predict-button');
    expect(predictBtn).toBeDisabled();
    expect(screen.getByRole('button', { name: /Retry/i })).toBeInTheDocument();
  });

  it('renders "Not reported" when latency_ms is omitted from response', async () => {
    const mockPredictionWithoutLatency = {
      model_version: '1.0.2',
      quality: 'good',
      probabilities: { good: 0.95, poor_or_reject: 0.05 },
      calibrated_confidence: 0.95,
      uncertainty: 0.12,
      ood_score: -10.5,
      decision: 'accept',
      quality_attributes: {
        artifact: 'none',
        clarity: 'high',
        field_definition: 'adequate'
      },
      feedback: ['Optimal macular center'],
      disclaimer: 'Technical image-quality assessment only.'
    };

    globalThis.fetch = vi.fn().mockImplementation((url: string) => {
      if (typeof url === 'string' && url.includes('/health/ready')) {
        return Promise.resolve({ ok: true, json: async () => ({ status: 'ready' }) } as Response);
      }
      return Promise.resolve({ ok: true, json: async () => mockPredictionWithoutLatency } as Response);
    });

    render(<App />);
    await waitFor(() => {
      expect(screen.getByText(/Model Ready/i)).toBeInTheDocument();
    });

    const fileInput = screen.getByTestId('file-input');
    const validFile = new File(['bytes'], 'fundus.jpg', { type: 'image/jpeg' });
    fireEvent.change(fileInput, { target: { files: [validFile] } });

    fireEvent.click(screen.getByTestId('predict-button'));

    await waitFor(() => {
      expect(screen.getByTestId('decision-badge')).toBeInTheDocument();
      expect(screen.getByText(/Not reported/i)).toBeInTheDocument();
    });
  });
});
