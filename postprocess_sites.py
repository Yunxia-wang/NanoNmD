"""
NanoNmD — Step 7: Map Signal Features to Genomic Sites
==========================================================
Merges per-read signal features with BAM-derived genomic coordinates
to produce one output file per sample:

    <SITE_FEATURE_DIR>/<sample>.all_kmer.rRNA.extract.sites.csv.gz

Columns in output:
    read_pos, CHROM, REF_POS, BASE, seq_mer, FLAG,
    site_strand, 0, 1, 2, 3, 4, 5, 6, 7, new_seq_mer

Usage (CLI):
    python postprocess_sites.py \\
        --bam_dir         /path/to/03_bam_signal_data \\
        --split_batch_dir /path/to/02_guppy_tombo_resquiggle_extract_data \\
        --site_feature_dir /path/to/04_site_feature

Or import and call process_all_samples() directly.
"""

import os
import argparse
import pandas as pd
import numpy as np
from pandas.errors import EmptyDataError
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Core processing
# ─────────────────────────────────────────────────────────────────────────────

def process_sample(sample_name: str, bam_dir: str, split_batch_dir: str,
                   site_feature_dir: str) -> bool:
    """
    Merge signal features with genomic coordinates for one sample.

    Parameters
    ----------
    sample_name      : Sample identifier (folder name inside bam_dir).
    bam_dir          : Root directory containing per-sample BAM metadata folders.
    split_batch_dir  : Root directory containing per-sample Guppy/feature folders.
    site_feature_dir : Output directory.

    Returns
    -------
    True on success, False on any handled error.
    """
    output_path = os.path.join(
        site_feature_dir,
        f'{sample_name}.all_kmer.rRNA.extract.sites.csv.gz'
    )
    if os.path.exists(output_path):
        tqdm.write(f'  SKIP {sample_name} — output already exists.')
        return True

    # ── File paths ────────────────────────────────────────────────────────────
    feat_path = os.path.join(
        split_batch_dir,
        f'{sample_name}_guppy',
        f'{sample_name}_guppy.feature.feature.tsv.gz',
    )
    bam_tsv_path = os.path.join(
        bam_dir,
        sample_name,
        f'{sample_name}.extract.sort.bam.tsv.gz',
    )

    # ── Load data ─────────────────────────────────────────────────────────────
    try:
        data_all = pd.read_csv(feat_path,    sep='\t', header=None)
        data_bam = pd.read_csv(bam_tsv_path, sep='\t')
    except FileNotFoundError as exc:
        tqdm.write(f'  ERROR {sample_name}: file not found — {exc}')
        return False
    except EmptyDataError:
        tqdm.write(f'  ERROR {sample_name}: empty file — {feat_path}')
        return False

    # ── Filter BAM rows with valid positions ──────────────────────────────────
    data_bam_sel = data_bam[
        (data_bam['READ_POS'] != '.') & (data_bam['REF_POS'] != '.')
    ].copy()

    if data_bam_sel.empty:
        tqdm.write(f'  WARN  {sample_name}: no valid positions in BAM metadata.')
        return False

    # ── Composite read ID: <read_name>|<read_pos>|N ───────────────────────────
    data_bam_sel['read_id'] = (
        data_bam_sel['#READ_NAME'].astype(str) + '|'
        + data_bam_sel['READ_POS'].astype(str) + '|N'
    )

    # ── Genomic site label: <chrom>_<ref_pos> ────────────────────────────────
    data_bam_sel['site_name'] = (
        data_bam_sel['CHROM'].astype(str) + '_'
        + data_bam_sel['REF_POS'].astype(str)
    )

    # ── Filter feature rows whose read_id appears in BAM ─────────────────────
    data_all_sel = data_all[data_all[0].isin(data_bam_sel['read_id'])].copy()
    data_all_sel['read_id'] = data_all_sel[0]

    if data_all_sel.empty:
        tqdm.write(f'  WARN  {sample_name}: no overlapping read IDs after merge.')
        return False

    # ── Merge ─────────────────────────────────────────────────────────────────
    data_combined = pd.merge(
        data_all_sel,
        data_bam_sel[['read_id', 'CHROM', 'REF_POS', 'BASE', 'FLAG', 'site_name']],
        on='read_id',
    )

    # ── Select and rename columns ─────────────────────────────────────────────
    # Signal feature columns (0–7) come from data_all.
    # Column 2 appears as both seq_mer and new_seq_mer (original design).
    data_final = data_combined[[
        'read_id', 'CHROM', 'REF_POS', 'BASE', 2, 'FLAG', 'site_name',
        0, 1, 2, 3, 4, 5, 6, 7, 2,
    ]].copy()

    data_final.columns = [
        'read_pos', 'CHROM', 'REF_POS', 'BASE', 'seq_mer', 'FLAG', 'site_strand',
        0, 1, 2, 3, 4, 5, 6, 7, 'new_seq_mer',
    ]

    # ── Save ──────────────────────────────────────────────────────────────────
    data_final.to_csv(output_path, sep=',', compression='gzip')
    tqdm.write(f'  OK    {sample_name} → {output_path}  ({len(data_final):,} rows)')
    return True


def process_all_samples(bam_dir: str, split_batch_dir: str,
                        site_feature_dir: str) -> None:
    """
    Iterate over all sample sub-folders found in bam_dir and process each.
    """
    os.makedirs(site_feature_dir, exist_ok=True)

    samples = sorted(
        entry.name
        for entry in os.scandir(bam_dir)
        if entry.is_dir()
    )

    if not samples:
        print(f'No sample folders found in: {bam_dir}')
        return

    print(f'Found {len(samples)} sample(s) to process.\n')

    success_count = 0
    for sample in tqdm(samples, desc='Processing samples', unit='sample'):
        tqdm.write(f'\n── {sample} ──')
        ok = process_sample(
            sample_name      = sample,
            bam_dir          = bam_dir,
            split_batch_dir  = split_batch_dir,
            site_feature_dir = site_feature_dir,
        )
        if ok:
            success_count += 1

    print(f'\nFinished: {success_count}/{len(samples)} samples processed successfully.')
    print(f'Output directory: {site_feature_dir}')


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='NanoNmD Step 7: Map signal features to genomic sites.'
    )
    p.add_argument('--bam_dir',         required=True,
                   help='Directory containing per-sample BAM metadata sub-folders '
                        '(e.g. 03_bam_signal_data)')
    p.add_argument('--split_batch_dir', required=True,
                   help='Directory containing per-sample Guppy/feature sub-folders '
                        '(e.g. 02_guppy_tombo_resquiggle_extract_data)')
    p.add_argument('--site_feature_dir', required=True,
                   help='Output directory for .csv.gz site-feature files '
                        '(e.g. 04_site_feature)')
    p.add_argument('--sample', default=None,
                   help='Process a single sample only (optional). '
                        'If omitted, all samples in bam_dir are processed.')
    return p.parse_args()


def main():
    args = parse_args()

    if args.sample:
        os.makedirs(args.site_feature_dir, exist_ok=True)
        process_sample(
            sample_name      = args.sample,
            bam_dir          = args.bam_dir,
            split_batch_dir  = args.split_batch_dir,
            site_feature_dir = args.site_feature_dir,
        )
    else:
        process_all_samples(
            bam_dir          = args.bam_dir,
            split_batch_dir  = args.split_batch_dir,
            site_feature_dir = args.site_feature_dir,
        )


if __name__ == '__main__':
    main()
