# Market and prior-art research

Scope: a short, evidence-linked scan done while productising this repo (October 2026). It is
not a market-sizing study. Where a claim comes from a vendor blog or from my own reading
rather than a primary source, it is marked.

## Who would use this

| User | Job to be done | What they need from a tool |
|---|---|---|
| Security operator / facilities team with fixed cameras | "Tell me which 5 minutes of 8 hours are worth watching" | A short review queue, few false alarms, a timestamp they can jump to |
| Small retail / site manager without a data team | Check overnight or weekend footage | Zero setup, drag and drop, plain-language output |
| Incident reviewer / supervisor | Document what was checked and the outcome | An exportable report with a reviewer sign-off section |
| Researcher / student | Try a baseline on a clip | Honest numbers, reproducible runs, no hidden tuning |

The honest primary fit today is the last two rows plus a lightweight triage use on a single
fixed camera. It is not a replacement for a video management system (VMS).

## What exists

| Project | What it does well | What it does not do (for this job) | Licence |
|---|---|---|---|
| [Frigate](https://github.com/blakeblackshear/frigate) (36k+ stars) | Mature local NVR with real-time object detection and Home Assistant integration | Detects known object classes; it is not an "anything unusual" detector. Its open issues include requests about false alerts and detection levels, e.g. "spray alarm & night time detection levels & cam disconnect alert" | MIT |
| [anomalib](https://github.com/open-edge-platform/anomalib) (6k+ stars) | Broad library of image anomaly algorithms with benchmarking | Aimed at image-level industrial inspection, not event segmentation in long surveillance video or reports for reviewers | Apache-2.0 |
| [MNAD](https://github.com/cvlab-yonsei/MNAD) (memory-guided normality, CVPR 2020) and [Future Frame Prediction](https://github.com/StevenLiuWen/ano_pred_cvpr2018) (CVPR 2018) | The standard research baselines on Ped2 / Avenue / ShanghaiTech | Research code: dataset-specific scripts, no serving layer, no thresholds for operators. GitHub shows no licence on either repo, so no code was reused here | none stated |
| Commercial VMS analytics | Integrated alerts, retention, multi-camera management | Closed, per-camera licensing, accuracy rarely published (not verified, general observation) | proprietary |

## What users and the literature complain about

- **False alarms and threshold choice.** A 2025 survey describes the sensitivity trade-off
  (catch most events but alarm often, or be strict and miss events) and says the balance is hard
  and application-specific: [arXiv 2508.14203](https://arxiv.org/pdf/2508.14203).
- **Domain shift.** The same survey notes that scene, angle and camera differences raise false
  positive rates, and that many methods are validated on single-scene datasets, which hides it.
- **Lab-to-field gap.** Reviews note that benchmark focus overlooks deployment issues such as
  device heterogeneity and resource limits: [arXiv 2405.10347](https://arxiv.org/pdf/2405.10347).
- **Inconsistent benchmark reporting.** Papers report micro-averaged (pool all test frames, one
  ROC) or macro-averaged (per-video AUC, then average) frame-level AUC, and some normalise
  scores per clip, so numbers are not directly comparable: see
  [arXiv 2008.12328](https://arxiv.org/pdf/2008.12328) and [arXiv 2010.07524](https://arxiv.org/pdf/2010.07524).
- **No explanations or reporting** in research code: outputs are score arrays and AUC.
- **Setup pain** in NVR-style tools (Frigate issues about detector and storage failures at
  startup).

## The gap this project can honestly own

Not "better accuracy" (the served model scores AUC 0.674 on Ped2; published methods report
roughly 90%+ on that set) but the layer around a weak-to-moderate scorer that makes it usable:

1. **A false-alarm budget instead of a magic number.** Flag at most X% of a clip, and only
   moments clearly above that clip's own baseline (median/MAD z-score), so the operator tunes
   workload, not an unintelligible reconstruction error.
2. **Event segments, not frame scores.** Consecutive flags merge into time-ranged segments with
   a peak and a plain-language explanation, ordered for review.
3. **A review loop and report.** Confirm / false-alarm per segment, exportable JSON, CSV,
   Markdown and a printable page with a sign-off section.
4. **Honest scope.** A domain-shift warning when a clip is clearly outside the training scene,
   and a README that states the benchmark number, its protocol (micro AUC, test set reused
   for selection) and its limits.
5. **Zero-friction start.** One container or one command, sample clips in the box, no account.

## What was reused

Only ideas, never code. Per-clip normalisation and robust baselines are common practice in the
literature above; event segmentation with gap merging and "budgeted" thresholds are standard
ideas, implemented from scratch in `src/incident_intel/analysis.py`. MNAD and Future Frame
Prediction carry no licence, so none of their code is copied. Frigate (MIT) and anomalib
(Apache-2.0) were read for product patterns only (review-oriented UI; benchmarking discipline).

## Not validated

The budget-mode heuristic (top 5% and at least 5 robust sigma above the median) has been checked
only on the three synthetic sample clips and unit tests. It has not been evaluated on Ped2,
Avenue or real footage. Treat it as an operating heuristic.

## Peers and lessons (product patterns)

Method and limits: I searched the web for the Product Hunt products the brief named. Several
could not be located under those names (withHuman, Fwrd, Basedash's "connector gallery"
preview), so I read the nearest verifiable analogues instead and say so. I did not read
Product Hunt comment threads or reviews, so there is no first-hand "user complaints" material
for these products; complaint evidence in the section above comes from the cited surveys and
Frigate's public issues. No text, assets or code were copied.

| Product (verified source) | What it does | Lesson for incident-intel | Status here |
|---|---|---|---|
| [HumanLayer](https://humanlayer.dev), [Velatir](https://hunted.space/product/velatir/launches/velatir-2), [Impri](https://www.hunted.space/dashboard/impri) (human-approval layers for AI agents) | Route a risky action to a person for approval, often through chat | Keep a human decision in the loop and record it | Done: Confirm / False alarm per segment, stored server-side, shown in reports, exported as labels CSV |
| [Stream Forwarder](https://www.producthunt.com/products/stream-forwarder-by-relbis-labs/makers), [alert_forwarder](https://codebot.readthedocs.io/en/latest/api/alert_forwarder.html) (stand-ins for the forwarding idea) | Pipe events to Slack, Discord or Telegram | Alerts should go where people already work | Done: operator-set `ALERT_WEBHOOK_URL` posts a Slack/Discord-style JSON message when an analysis flags segments. Telegram needs a relay; not built |
| [Basedash](https://docs.basedash.com/changelog/october-3) (data-source onboarding) | Guided connection flow | Show what the user will get before asking for commitment | Done: bundled samples plus a real example on the landing page |
| Product Hunt launches generally | Open-source and self-host positioning, API/MCP access, `llms.txt` | Make it easy for people and tools to understand and adopt | Done: self-host Docker quickstart, OpenAPI at `/docs`, `/llms.txt`. MCP server: not built |
| Kerno, CodeCrab, Afterparty, Inferock, Actually Asked, ReSO AI | Not located or not verified | None claimed | n/a |

What to borrow, ranked by user value over effort (what I built, in this order):

1. Human review loop with exportable labels (high value, low effort). Labels are the raw material for any future calibration. The model does not learn from them today; the only automatic effect is a plain hint ("most reviewed flags were false alarms, lower the alert budget").
2. Webhook alerts (high value for monitoring use, low effort).
3. Shareable report links and API docs plus `llms.txt` (medium value, low effort).
4. Real-example gallery (medium value, done through samples and the landing page).

Not built, deliberately: automatic threshold learning from feedback (too little data to do honestly), Telegram direct integration, an MCP server, and credit-back style guarantees (the model gives no accuracy guarantee to refund against).

What to avoid: claims of accuracy beyond Ped2, fake logos or testimonials, dark patterns around sign-up (there is no sign-up), and letting a hosted instance collect video (uploads are deleted after scoring).
