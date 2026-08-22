"""PlumeChaser: open autonomous methane super-emitter monitoring.

Pipeline (see docs/RUNBOOK.md):
    T0 ingest -> T1 detect -> T2 verify -> T3 cue -> T4 MBMP+IME
    -> T5 attribution context -> T6 dossiers/replay bundles.

Scientific foundations:
    Varon et al. 2021, AMT 14, 2771-2793 (MBMP retrieval + IME method)
    Schuit et al. 2023, ACP 23, 9071-9098 (automated TROPOMI detection)
    Gorrono et al. 2023, AMT 16, 89-117 (Sentinel-2 detection limits)

License: MIT (code). Third-party data carry their own licenses; see README.
"""

__version__ = "0.1.0"
