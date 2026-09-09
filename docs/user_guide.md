# RetinaGuard-QA Operator User Guide

## Step-by-Step Operator Workflow

1. **Initiate Scan Inspection**:
   - Access the web interface at `http://localhost:8000`.
   - Drag and drop the captured retinal fundus photograph into the ingestion box.
2. **Review Triage Action**:
   - **Green (`ACCEPT`)**: Image satisfies technical quality criteria. Proceed to diagnostic AI or clinical review.
   - **Yellow (`USABLE_WITH_WARNING`)**: Minor peripheral defect present. The macula and optic disc remain interpretable.
   - **Red (`RECAPTURE_WITH_GUIDANCE`)**: Serious optical/physical defect detected (e.g. defocus blur, underexposure). Read the numbered corrective steps and adjust camera settings before retaking the image.
   - **Purple (`MANUAL_REVIEW`)**: High epistemic uncertainty detected near decision boundary. Requires human operator review.
   - **Rose (`UNSUPPORTED_OOD`)**: Non-fundus image, severely corrupt file, or unfamiliar optical modality detected.
3. **Export Audit Certification**:
   - Click **Download PDF Audit Report** to generate a timestamped, signed PDF quality assessment for electronic medical records (EMR) integration.
