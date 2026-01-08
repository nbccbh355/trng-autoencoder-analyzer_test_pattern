
# TRNG Autoencoder Analyzer – Test Pattern Extension
## About this fork

This repository is a fork of the original **trng-autoencoder-analyzer** project.
It is used to experiment with raw TRNG test patterns prior to encoding and
downstream analysis.

The additional `trng.py` script focuses on pattern-level preprocessing and
inspection, providing an intuitive way to observe entropy behavior before
feeding data into statistical tests or autoencoder-based pipelines.

## Purpose

- Explore raw TRNG bitstream behavior at the pattern level
- Perform lightweight SP800-90B–style health checks (e.g. repetition and
  proportion tests)
- Generate example outputs to support exploratory analysis and learning

This fork is intended for research and educational use.

## Quick usage

```bash
python trng.py --input sample.bin --out out --no-ae

## Quick usage
```bash
python trng.py --input sample.bin --out out --save-heatmap --no-ae


Technical Article:  
AI-Assisted TRNG Entropy Analysis Using SP800-90B and Autoencoder Residual Mapping  
https://medium.com/@ace.lin0121/ai-assisted-trng-entropy-analysis-using-sp800-90b-and-autoencoder-residual-mapping-cdf2ca6e3cb1

TRNG Autoencoder Analysis – Documentation
========================================

This repository provides a public technical presentation (PPTX file)
describing a conceptual framework for evaluating True Random Number
Generator (TRNG) entropy quality using NIST SP800-90B principles and
an Autoencoder-based residual inspection approach.

Purpose
-------

The purpose of this repository is to make the methodology explanation
and related research notes publicly accessible.  
This allows external verification, timestamped publication, and a
permanent public record of the work.

Repository Contents
-------------------

AE_SP80090B_Safe_v2.pptx  
A technical slide deck covering:
- Motivation for applying machine learning techniques to TRNG analysis
- Summary of SP800-90B entropy-related concepts  
  (collision tests, symbol frequency, estimator considerations)
- A conceptual Autoencoder residual analysis idea for identifying
  potential patterns or anomalies in TRNG output data
- Discussion of how statistical and AI-based inspection can complement
  each other in TRNG quality evaluation

README.md  
This file.

Current Status
--------------

This repository currently provides documentation only.
No scripts, datasets, or executable code are included at this time.

Planned Additions
-----------------

Additional materials may be published in the future, such as:
- Extended documentation or notes
- Supporting figures or examples

Versioning
----------

v1.0.0 – Initial public release (documentation only)

Contact
-------

For questions or clarifications, please use GitHub Issues.
