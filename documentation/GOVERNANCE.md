# Governance

**Principle:** MeghPrahari is decision support. It drafts; authorised humans decide. It fails closed.

| Control | Where enforced |
|---|---|
| Shadow mode default (`shadow_mode: true`): CAP status `Exercise`, scope `Restricted`, note added; never `Actual` | `alerts.draft_alert`, `settings.yaml` |
| Replay/hindcast always forces shadow and sets `hindcast=true` | `worker.run_cycle` |
| Alerts are created as `draft`; publishing needs approval by the `approver` role | `api.py`, `governance.is_approved` |
| Four-eyes: level 3 (ACT_NOW) needs two distinct approvers | `governance.approvals_needed` |
| Approved CAP content is hash-locked; publish is blocked if the stored XML no longer matches | `api.publish` (`cap_sha256`) |
| Only calibrated models can drive alerts; uncalibrated models raise | `model.predict`, DB CHECK on `model_version` |
| Model approval by a scientist other than the trainer (separation of duties) | `api.approve_model` |
| Model files load only if SHA-256 equals the approved registration (pickle safety) | `model.load_model` |
| Tamper-evident audit: hash chain + append-only triggers; `/api/audit/verify` | `governance`, `database/init.sql` |
| Role-based access (admin, scientist, approver, operator, volunteer, auditor) | `governance.PERMISSIONS` |
| Start-up refuses unset configuration (`CHANGE_ME`) | `settings.load_settings` |
| Data minimisation: no phone numbers/contacts stored; dissemination through the authority's gateway | `database/init.sql` |

## Go-live gates (do not skip)
1. Written agreement with the issuing authority (SDMA/DDMA/IMD/NCMRWF) on roles, sender identifier, CAP profile and legal responsibility.
2. Data licences confirmed for every source (MOSDAC and NCMRWF terms are research-use unless licensed).
3. At least one full monsoon in shadow mode with published verification (POD/FAR/CSI, reliability, skill vs pysteps and vs persistence).
4. Cost–loss values (`action_cost`, `loss`) and level thresholds agreed with the authority and recorded.
5. CAP output validated against the OASIS 1.2 XSD and the authority's profile; delivery tested end to end with `Exercise` messages.
   *Status:* the OASIS CAP 1.2 XSD check is automated (`tests/test_cap_schema.py`); the authority's profile and
   end-to-end delivery are still open.
6. Restore test of the database backup; audit chain verified after restore.
7. Incident process: every false alarm and miss reviewed within a set time; results appended to the verification log.
8. Only then switch `shadow_mode: false`, under change control, by an admin with a second person witnessing.
