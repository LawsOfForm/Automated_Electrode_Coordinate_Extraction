def construct_baseline_coordinate_table(tables, stim_folders=('sham', 'active'),
                                        on_conflict='first'):
    """Build the intended-montage table from the SimNIBS result pickles.

    Searches every folder in `stim_folders` under

        <root>/<stim>/02-ANALYSIS/**/*.pkl

    The previous version searched 'sham' only, so subjects whose optimisation
    was stored under 'active' had no baseline and were silently dropped from
    every downstream comparison.

    CONFLICTS
    ---------
    The dictionary key is Exp_condition_subject, which does NOT include the
    stimulation folder, so the same key can now arrive from both 'sham' and
    'active'. Overwriting one with the other silently -- which is what a plain
    dict assignment does -- would make the baseline depend on directory
    iteration order. Instead:

      * identical coordinates (within 0.01 mm): keep one, no message. This is
        the expected case for a montage that is physically the same and only
        differs in the current waveform;
      * differing coordinates: report the key, the distance, and both sources,
        then resolve by `on_conflict`:
            'first'  keep the folder listed earliest in stim_folders
            'skip'   drop the key entirely, so no correction is attempted
                     against an ambiguous target

    The returned table gains a `stim` column recording which folder each
    baseline came from, so a conflict can be traced afterwards.
    """
    root_table = '/media/MeMoSLAP_Mesh2/PDF_Report_Generation'

    pickle_files = []
    per_folder = {}
    for stim in stim_folders:
        hits = glob.glob(os.path.join(root_table, stim, '02-ANALYSIS',
                                      '**', '*.pkl'), recursive=True)
        per_folder[stim] = len(hits)
        pickle_files += [(stim, h) for h in hits]

    print("\n  Baseline pickles found:")
    for stim, n in per_folder.items():
        print(f"    {stim:<8} {n:>5}")
    if not pickle_files:
        print(f"    none under {root_table}/<{'|'.join(stim_folders)}>/02-ANALYSIS")
        print("    Check the path and the folder names.")

    dict_data, source, conflicts, skipped = {}, {}, [], []
    pkl_iter = (tqdm(pickle_files, desc="Loading baseline pickles", unit="file")
                if TQDM_AVAILABLE else pickle_files)

    for stim, file in pkl_iter:
        folder_str = os.path.basename(os.path.dirname(file))
        parts = folder_str.split('_')
        if len(parts) < 3:
            skipped.append((file, f"folder name '{folder_str}' is not "
                                  f"Exp_condition_subject"))
            continue
        Exp, tgt, sub_id = parts[0], parts[1], parts[2]
        sub = f'sub-{sub_id}'
        try:
            with open(file, 'rb') as f:
                data = pickle.load(f)
            key_inner = list(data[2].keys())[0]
            coords = {
                'anode':    list(data[1]),
                'cathode1': list(data[2][key_inner][0]),
                'cathode2': list(data[2][key_inner][1]),
                'cathode3': list(data[2][key_inner][2]),
            }
        except Exception as exc:                                # noqa: BLE001
            skipped.append((file, f"{type(exc).__name__}: {exc}"))
            continue

        key = f'{Exp}_{tgt}_{sub}'
        if key in dict_data:
            prev = dict_data[key]
            d = max(float(np.linalg.norm(np.array(prev[e]) - np.array(coords[e])))
                    for e in ('anode', 'cathode1', 'cathode2', 'cathode3'))
            if d <= 0.01:
                continue                       # same montage in both folders
            conflicts.append((key, source[key], stim, d))
            if on_conflict == 'skip':
                dict_data.pop(key, None)
                source.pop(key, None)
                continue
            continue                           # 'first': keep what we have
        dict_data[key] = coords
        source[key] = stim

    if conflicts:
        print(f"\n  CONFLICT: {len(conflicts)} subject(s) have a baseline in "
              f"more than one stimulation folder with DIFFERENT coordinates.")
        print(f"  Resolution: {'kept the first folder listed' if on_conflict=='first' else 'dropped entirely'}.")
        for key, first, second, d in conflicts[:20]:
            print(f"    {key:<28} {first} vs {second}   max difference {d:7.1f} mm")
        if len(conflicts) > 20:
            print(f"    ... and {len(conflicts)-20} more")
        print("  These are worth resolving at source: the correction step will")
        print("  match against whichever montage was kept.")

    if skipped:
        print(f"\n  {len(skipped)} pickle(s) could not be read:")
        for f, why in skipped[:10]:
            print(f"    {os.path.basename(os.path.dirname(f))}: {why}")
        if len(skipped) > 10:
            print(f"    ... and {len(skipped)-10} more")

    if not dict_data:
        raise SystemExit(
            "\n  No baseline coordinates could be loaded. Every downstream "
            "correction depends on them, so stopping here rather than "
            "producing an empty table.")

    df_template = pd.DataFrame.from_dict(dict_data, orient='index').reset_index()
    df_template[['exp', 'condition', 'subject']] = (
        df_template['index'].str.split('_', expand=True)
    )
    df_template['stim'] = df_template['index'].map(source)
    df_template['session'] = 'ses-baseline'
    df_template['run'] = 'run-baseline'
    df_template.to_csv(os.path.join(tables, 'baseline_coordinate_table.csv'),
                       index=False)

    print(f"\n  Baseline table: {len(df_template)} subject/condition entries")
    for stim, n in df_template['stim'].value_counts().items():
        print(f"    from {stim:<8} {n:>5}")
    print(f"    unique subjects {df_template['subject'].nunique()}\n")
    return df_template
