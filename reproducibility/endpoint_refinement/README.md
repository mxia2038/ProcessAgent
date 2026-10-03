# Offline endpoint refinement

This package contains ten numerical endpoint records, their historical refinement results and reconstructed solve counts, the public-property process model, and an offline verification script. It contains no agent conversations, prompts copied from conversations, account settings or API credentials.

With Python 3.13.5 and the dependencies in requirements-offline.txt, run from this directory:

```
python verify_endpoints.py
```

The script verifies the supplied endpoint objective and strict feasibility, repeats SLSQP refinement and tests strict feasibility at the refined boundary. Results are written to verification_results/endpoint_revision_checks.json. No API calls are made.

All ten supplied endpoints improve by 0.58-1.59%. Recorded and repeated refined objectives agree within 0.000033 kg steam per tonne of dry NaOH. Exact historical solve counts differ: median refinement count 31 versus 30 on repetition; direct SLSQP from the common start takes 20 versus 36 solves. The strict sensitivity test gives median 30.5 and direct count 25.

Endpoint selection and agent solve counts are supplied experimental summaries, not reconstructed from raw conversations by this package. Agent counts are upper-bound estimates. This package verifies numerical refinement of the supplied endpoints; it does not reproduce LLM generations, independently audit termination messages, or establish that these endpoints are the best points in the original histories. Full conversations are retained locally and are not included in this release.

Both optimisers and agent protocols use the same process equations. Numerical solvers receive continuous constraint residuals; the agent validator returns a directional explanation for the first violated constraint and exposes the objective only after successful validation. The resolved historical LLM snapshot, effective sampling settings and numerical library versions are unavailable.
