%% ============================================================
%  CONFIGURATION
%% ============================================================
%
%  CHANGES vs. the previous version (all marked [FIX] / [NEW] below):
%    [FIX] Missing petra run-02 no longer crashes the unzip step.
%    [FIX] Multiple dir() matches no longer crash; first is used and
%          every ignored match is written to the log.
%    [FIX] The coregistration skip-check now requires BOTH rsub-run-01
%          and rsub-run-02, so a half-finished session is redone.
%    [FIX] Unzipping is wrapped in try/catch, so one bad archive cannot
%          abort a run spanning hundreds of subjects.
%    [NEW] Subject-level skip: a subject whose files are all unzipped /
%          all sessions coregistered is skipped without touching disk.
%    [NEW] Counters and an end-of-run summary.
%
%% ============================================================

RUN_UNZIP          = true;
RUN_COREGISTRATION = true;
UNZIP_MODE         = 'new_only';   % 'new_only' or 'overwrite'
COREG_MODE         = 'new_only';   % 'new_only' or 'overwrite'

% ── PROJECT FILTER ───────────────────────────────────────────
%  The project is defined by the FIRST digit of the subject
%  number, e.g.:
%    sub-9001  →  project 9
%    sub-8042  →  project 8
%    sub-7103  →  project 7
%
%  Set PROJECT_FILTER to a list of project IDs (integers) to
%  process only those projects.  Leave it EMPTY ([]) to process
%  ALL subjects regardless of project.
%
%  Examples:
%    PROJECT_FILTER = [9];        % only project 9
%    PROJECT_FILTER = [8, 9];     % projects 8 and 9
%    PROJECT_FILTER = [];         % no filter — all subjects
% ─────────────────────────────────────────────────────────────
PROJECT_FILTER = [];   % <-- edit here

%% ============================================================
%  PATHS
%% ============================================================

path                = "/media/MeMoSLAP_Subjects/SUBJECTS_XNAT";
coregistration_path = "/media/MeMoSLAP_Subjects/derivatives/automated_electrode_extraction";
log_file            = 'Logfile_Coregistration.txt';

%% ============================================================
%  TMP / SCRATCH FOLDER
%  All temporary files (logs, batch .mat files, SPM journals)
%  are written here. Delete the whole folder to clean up.
%% ============================================================

% Place _tmp next to this script, wherever it lives
script_dir = fileparts(mfilename('fullpath'));
tmp_dir    = fullfile(script_dir, '_tmp');

if ~exist(tmp_dir, 'dir')
    mkdir(tmp_dir);
    fprintf('Created tmp folder: %s\n', tmp_dir);
end

% Redirect MATLAB / SPM journal and temp files into tmp_dir
setenv('TMPDIR', tmp_dir);          % UNIX tmp override for external tools
orig_tmpdir = tempdir();            % keep original for reference only

% Redirect SPM's own job-manager journal file
try
    spm_get_defaults('cmdline', true);   % suppress SPM GUI popups
catch
    % SPM not yet on path at config time — that's fine, set later
end

% Start a MATLAB diary inside tmp_dir so fprintf output is also captured
diary_file = fullfile(tmp_dir, ['diary_', datestr(now,'yyyymmdd_HHMMSS'), '.txt']);
diary(diary_file);
diary on;

fprintf('=== Coregistration script started: %s ===\n', datestr(now));
fprintf('Tmp/scratch folder : %s\n', tmp_dir);
fprintf('Diary log          : %s\n', diary_file);

%% ============================================================
%  SETUP
%% ============================================================

dummy_folder_list   = {'ses-base','ses-1','ses-2','ses-3','ses-4'};
all_session         = {'sesbase','ses1','ses2','ses3','ses4'};
valid_session_pattern = '^ses-(base|[1-4])$';

% Main log file also lives in tmp_dir (easy to find and delete)
log_path = fullfile(tmp_dir, log_file);
fileID = fopen(log_path, 'w');
if fileID == -1
    error('Could not open log file: %s', log_path);
end
fprintf('Log file           : %s\n\n', log_path);

% [NEW] Counters for the end-of-run summary
n_sub_unzip_skipped = 0;
n_sub_coreg_skipped = 0;
n_unzip_done        = 0;
n_unzip_failed      = 0;
n_coreg_done        = 0;
n_coreg_skipped     = 0;
n_coreg_failed      = 0;

%% ============================================================
%  STEP 0 — CLEAN UP MISMATCHED UNZIPPED FILES
%  For every subject folder in the derivatives directory,
%  check that every .nii file inside unzipped/ actually
%  belongs to that subject. Delete any that do not.
%
%  NOTE: coregistered outputs are named rsub-* and therefore do
%  not match '^(sub-\d+)_'. They are reported as unparseable and
%  left untouched, which is intended.
%% ============================================================

fprintf('\n===== CHECKING UNZIPPED FOLDERS FOR MISMATCHES =====\n');

deriv_dirs = dir(coregistration_path);
deriv_dirs = deriv_dirs([deriv_dirs.isdir]);
deriv_dirs = deriv_dirs(~ismember({deriv_dirs.name},{'.','..'}));

for d = 1:length(deriv_dirs)
    subject_folder_name = deriv_dirs(d).name;

    % Only process folders that look like subject folders
    if ~(length(subject_folder_name) == 7 || length(subject_folder_name) == 8)
        continue
    end

    unzipped_dir = fullfile(coregistration_path, subject_folder_name, 'unzipped');
    if ~exist(unzipped_dir, 'dir')
        continue
    end

    nii_files = dir(fullfile(unzipped_dir, '*.nii'));

    if isempty(nii_files)
        fprintf('  %s/unzipped — empty, skipping\n', subject_folder_name);
        continue
    end

    n_mismatch = 0;

    for f = 1:length(nii_files)
        fname = nii_files(f).name;

        % [FIX] Skip coregistered outputs quietly instead of warning about
        % every single one of them on every run.
        if strncmp(fname, 'rsub-', 5)
            continue
        end

        % Every BIDS .nii starts with the subject label, e.g. sub-0900_...
        tokens = regexp(fname, '^(sub-\d+)_', 'tokens');

        if isempty(tokens)
            fprintf('    WARNING: Cannot parse subject from filename: %s — skipping\n', fname);
            fprintf(fileID, 'WARNING: Cannot parse subject from: %s/%s\n', ...
                    subject_folder_name, fname);
            continue
        end

        file_subject = tokens{1}{1};   % e.g. 'sub-1103'

        if ~strcmp(file_subject, subject_folder_name)
            full_path = fullfile(unzipped_dir, fname);
            fprintf('    MISMATCH: %s belongs to %s — deleting\n', ...
                    fname, file_subject);
            fprintf(fileID, 'MISMATCH DELETED: %s (found in %s/unzipped)\n', ...
                    fname, subject_folder_name);
            delete(full_path);
            n_mismatch = n_mismatch + 1;
        end
    end

    if n_mismatch > 0
        fprintf('  %s/unzipped — %d mismatched file(s) deleted\n', ...
                subject_folder_name, n_mismatch);
    end
end

fprintf('===== MISMATCH CHECK COMPLETE =====\n\n');

%% ============================================================
%  BUILD FILE LIST
%
%  [FIX] Every dir() result is passed through pick_one(), which
%  collapses multiple matches to the first one and logs the rest.
%  Previously two matching files made `T1.name` throw
%  "Expected one output ... but there were 2 results".
%% ============================================================

[SubFolderDir, SubFolderNames] = get_subdir(path);

valid = cellfun(@(n) length(n) == 7 || length(n) == 8, SubFolderNames);
SubFolderNames = SubFolderNames(valid);
SubFolderDir   = SubFolderDir(valid);

filelist = struct();

for subfolder_subject = 1:length(SubFolderNames)
    sub_path = fullfile(SubFolderDir{subfolder_subject}, SubFolderNames{subfolder_subject});
    [SesFolderDir, SesFolderNames] = get_subdir(sub_path);
    sub_name = erase(SubFolderNames{subfolder_subject}, "-");

    fprintf('Subject #%d: %s\n', subfolder_subject, SubFolderNames{subfolder_subject});

    % Filter invalid session folders
    valid_ses   = cellfun(@(n) ~isempty(regexp(n, valid_session_pattern, 'once')), SesFolderNames);
    invalid_ses = SesFolderNames(~valid_ses);
    for k = 1:length(invalid_ses)
        fprintf('  WARNING: Skipping unexpected session folder: %s/%s\n', ...
                SubFolderNames{subfolder_subject}, invalid_ses{k});
        fprintf(fileID, 'WARNING: Skipping unexpected session folder: %s/%s\n', ...
                SubFolderNames{subfolder_subject}, invalid_ses{k});
    end
    SesFolderNames = SesFolderNames(valid_ses);
    SesFolderDir   = SesFolderDir(valid_ses);

    filelist.(sub_name).Session_list  = SesFolderNames;
    % Store the RAW folder name (e.g. 'sub-0900') so paths are always correct
    filelist.(sub_name).raw_name      = SubFolderNames{subfolder_subject};

    for subfolder_session = 1:length(SesFolderNames)
        ses_name = erase(SesFolderNames{subfolder_session}, "-");
        ses_path = fullfile(SesFolderDir{subfolder_session}, SesFolderNames{subfolder_session});
        lbl      = [SubFolderNames{subfolder_subject}, ' ', SesFolderNames{subfolder_session}];

        filelist.(sub_name).Session.(ses_name).T1 = pick_one( ...
            dir(fullfile(ses_path, '**/*mprage_T1w.nii.gz')),        [lbl ' T1'],         fileID);
        filelist.(sub_name).Session.(ses_name).Petra_Pre = pick_one( ...
            dir(fullfile(ses_path, '**/*petra_run-01_PDw.nii.gz')),  [lbl ' petra run-01'], fileID);
        filelist.(sub_name).Session.(ses_name).Petra_Post = pick_one( ...
            dir(fullfile(ses_path, '**/*petra_run-02_PDw.nii.gz')),  [lbl ' petra run-02'], fileID);
    end
end

%% ============================================================
%  PROJECT FILTER
%% ============================================================

All_Subjects = fieldnames(filelist);

if ~isempty(PROJECT_FILTER)
    keep = true(size(All_Subjects));
    for s = 1:length(All_Subjects)
        raw = filelist.(All_Subjects{s}).raw_name;   % e.g. 'sub-0900'
        num_str = regexp(raw, '^sub-(\d+)$', 'tokens', 'once');
        if isempty(num_str)
            keep(s) = false;   % malformed name — exclude
            continue
        end
        project_id = str2double(num_str{1}(1));
        keep(s) = ismember(project_id, PROJECT_FILTER);
    end

    excluded     = All_Subjects(~keep);
    All_Subjects = All_Subjects(keep);

    fprintf('Project filter active: [%s]\n', ...
            strjoin(arrayfun(@num2str, PROJECT_FILTER, 'UniformOutput', false), ', '));
    fprintf('  Subjects kept    : %d\n', length(All_Subjects));
    fprintf('  Subjects excluded: %d\n', length(excluded));
    for e = 1:length(excluded)
        fprintf('    excluded: %s\n', filelist.(excluded{e}).raw_name);
        fprintf(fileID, 'PROJECT FILTER excluded: %s\n', filelist.(excluded{e}).raw_name);
    end
    fprintf('\n');
else
    fprintf('Project filter: none — processing all %d subjects\n\n', length(All_Subjects));
end

%% ============================================================
%  VALIDATE SUBJECTS
%% ============================================================

valid_subjects = true(size(All_Subjects));

for sub = 1:length(All_Subjects)
    subj = All_Subjects{sub};

    if ~isfield(filelist.(subj), 'Session') || ...
       ~isfield(filelist.(subj).Session, 'sesbase')
        fprintf('  WARNING: %s has no valid ses-base, skipping.\n', subj);
        fprintf(fileID, 'WARNING: %s has no valid ses-base, skipping.\n', subj);
        valid_subjects(sub) = false;
        continue
    end

    if isempty(filelist.(subj).Session.sesbase.T1)
        fprintf('  WARNING: %s has no T1 in ses-base, skipping.\n', subj);
        fprintf(fileID, 'WARNING: %s has no T1 in ses-base, skipping.\n', subj);
        valid_subjects(sub) = false;
        continue
    end

    petra_sessions = {'ses1','ses2','ses3','ses4'};
    has_petra = false;
    for k = 1:length(petra_sessions)
        ses = petra_sessions{k};
        if isfield(filelist.(subj).Session, ses) && ...
           ~isempty(filelist.(subj).Session.(ses).Petra_Pre)
            has_petra = true;
            break
        end
    end

    if ~has_petra
        fprintf('  WARNING: %s has no valid Petra sessions, skipping.\n', subj);
        fprintf(fileID, 'WARNING: %s has no valid Petra sessions, skipping.\n', subj);
        valid_subjects(sub) = false;
    end
end

All_Subjects = All_Subjects(valid_subjects);
fprintf('\n%d/%d subjects passed validation.\n\n', ...
        sum(valid_subjects), length(valid_subjects));

%% ============================================================
%  STEP 1 — UNZIPPING
%
%  [NEW] Subject-level skip via subject_unzip_done().
%  [FIX] Petra_Post is checked for emptiness before use.
%  [FIX] Each gunzip is wrapped in try/catch.
%% ============================================================

if RUN_UNZIP
    fprintf('\n===== UNZIPPING =====\n');

    for sub = 1:length(All_Subjects)
        subj     = All_Subjects{sub};
        sub_raw  = filelist.(subj).raw_name;   % e.g. 'sub-0900'

        % ---- [NEW] SUBJECT-LEVEL SKIP -----------------------------
        if strcmp(UNZIP_MODE, 'new_only') && ...
           subject_unzip_done(filelist, subj, coregistration_path, all_session)
            fprintf('  %s — everything already unzipped, skipping subject\n', sub_raw);
            n_sub_unzip_skipped = n_sub_unzip_skipped + 1;
            continue
        end

        for session_num = 1:length(all_session)
            ses = all_session{session_num};

            if ~isfield(filelist.(subj).Session, ses)
                continue
            end

            %% --- T1 (sesbase only) ---
            if strcmp(ses, 'sesbase')
                if ~isempty(filelist.(subj).Session.(ses).T1)
                    T1_folder = filelist.(subj).Session.(ses).T1.folder;
                    T1_name   = filelist.(subj).Session.(ses).T1.name;
                    T1_path   = fullfile(T1_folder, T1_name);
                    T1_out    = fullfile(coregistration_path, sub_raw, 'unzipped', T1_name(1:end-3));

                    if strcmp(UNZIP_MODE, 'overwrite') || ~exist(T1_out, 'file')
                        fprintf('  Unzipping T1: %s %s\n', sub_raw, ses);
                        try
                            gunzip(T1_path, fullfile(coregistration_path, sub_raw, 'unzipped'));
                            n_unzip_done = n_unzip_done + 1;
                        catch ME
                            fprintf('    ERROR unzipping T1 %s — %s\n', sub_raw, ME.message);
                            fprintf(fileID, 'ERROR unzipping T1 %s — %s\n', sub_raw, ME.message);
                            n_unzip_failed = n_unzip_failed + 1;
                        end
                    else
                        fprintf('  T1 already unzipped, skipping: %s %s\n', sub_raw, ses);
                    end
                else
                    fprintf('  T1 not found: %s %s\n', sub_raw, ses);
                    fprintf(fileID, 'T1 not found: %s %s\n', sub_raw, ses);
                end

            %% --- Petra Pre/Post (ses1-ses4) ---
            else
                Pre_entry  = filelist.(subj).Session.(ses).Petra_Pre;
                Post_entry = filelist.(subj).Session.(ses).Petra_Post;

                if isempty(Pre_entry)
                    fprintf('  Petra not found: %s %s\n', sub_raw, ses);
                    fprintf(fileID, 'Petra not found: %s %s\n', sub_raw, ses);
                    continue
                end

                out_dir  = fullfile(coregistration_path, sub_raw, 'unzipped');
                Pre_path = fullfile(Pre_entry.folder, Pre_entry.name);
                Pre_out  = fullfile(out_dir, Pre_entry.name(1:end-3));

                % [FIX] run-02 may legitimately be missing. Report it and
                % carry on with run-01 instead of throwing on .folder of an
                % empty struct, which used to abort the entire script.
                if isempty(Post_entry)
                    fprintf('  WARNING: %s %s has run-01 but NO run-02 — ', sub_raw, ses);
                    fprintf('unzipping run-01 only (session cannot be coregistered)\n');
                    fprintf(fileID, 'WARNING: %s %s missing petra run-02\n', sub_raw, ses);
                end

                if strcmp(UNZIP_MODE, 'overwrite') || ~exist(Pre_out, 'file')
                    fprintf('  Unzipping Petra: %s %s\n', sub_raw, ses);
                    try
                        gunzip(Pre_path, out_dir);
                        if ~isempty(Post_entry)
                            gunzip(fullfile(Post_entry.folder, Post_entry.name), out_dir);
                        end
                        n_unzip_done = n_unzip_done + 1;
                    catch ME
                        fprintf('    ERROR unzipping Petra %s %s — %s\n', sub_raw, ses, ME.message);
                        fprintf(fileID, 'ERROR unzipping Petra %s %s — %s\n', sub_raw, ses, ME.message);
                        n_unzip_failed = n_unzip_failed + 1;
                    end
                else
                    fprintf('  Petra already unzipped, skipping: %s %s\n', sub_raw, ses);
                end
            end
        end
    end

else
    fprintf('\n===== UNZIPPING SKIPPED =====\n');
end

%% ============================================================
%  STEP 2 — COREGISTRATION
%
%  [NEW] Subject-level skip: if every session that has data is
%        already coregistered, the subject is skipped without any
%        dir() calls or SPM startup.
%  [FIX] "already coregistered" now requires BOTH rsub-run-01 and
%        rsub-run-02. Previously a session whose run-02 reslice
%        had failed was treated as complete and never retried.
%% ============================================================

if RUN_COREGISTRATION
    fprintf('\n===== COREGISTRATION =====\n');

    coreg_sessions   = {'ses1','ses2','ses3','ses4'};
    coreg_dummy_list = {'ses-1','ses-2','ses-3','ses-4'};

    for sub = 1:length(All_Subjects)
        subj    = All_Subjects{sub};
        sub_raw = filelist.(subj).raw_name;   % e.g. 'sub-0900'

        % ---- [NEW] SUBJECT-LEVEL SKIP -----------------------------
        if strcmp(COREG_MODE, 'new_only')
            pending = {};
            for session_num = 1:length(coreg_sessions)
                ses = coreg_sessions{session_num};
                if ~isfield(filelist.(subj).Session, ses) || ...
                   isempty(filelist.(subj).Session.(ses).Petra_Pre) || ...
                   isempty(filelist.(subj).Session.(ses).Petra_Post)
                    continue
                end
                if ~session_coreg_done(coregistration_path, sub_raw, coreg_dummy_list{session_num})
                    pending{end+1} = ses;  %#ok<SAGROW>
                end
            end
            if isempty(pending)
                fprintf('\n  Subject: %s — all sessions already coregistered, skipping\n', sub_raw);
                n_sub_coreg_skipped = n_sub_coreg_skipped + 1;
                continue
            end
            fprintf('\n  Subject: %s  (%d session(s) pending: %s)\n', ...
                    sub_raw, length(pending), strjoin(pending, ', '));
        else
            fprintf('\n  Subject: %s  (COREG_MODE = overwrite)\n', sub_raw);
        end

        for session_num = 1:length(coreg_sessions)
            ses = coreg_sessions{session_num};

            if ~isfield(filelist.(subj).Session, ses) || ...
               isempty(filelist.(subj).Session.(ses).Petra_Pre) || ...
               isempty(filelist.(subj).Session.sesbase.T1)
                fprintf('    No valid data for %s %s, skipping\n', sub_raw, ses);
                fprintf(fileID, 'No valid data for %s %s, skipping\n', sub_raw, ses);
                continue
            end

            % [FIX] cheap check before any dir() calls
            if strcmp(COREG_MODE, 'new_only') && ...
               session_coreg_done(coregistration_path, sub_raw, coreg_dummy_list{session_num})
                fprintf('    Already coregistered, skipping: %s %s\n', sub_raw, ses);
                n_coreg_skipped = n_coreg_skipped + 1;
                continue
            end

            % --- Build paths using sub_raw throughout ---
            T1_unzip   = dir(fullfile(coregistration_path, sub_raw, 'unzipped', ...
                             [sub_raw, '_ses-base_acq-mprage_T1w.nii']));
            Pre_unzip  = dir(fullfile(coregistration_path, sub_raw, 'unzipped', ...
                             [sub_raw, '_', coreg_dummy_list{session_num}, '_acq-petra_run-01_PDw.nii']));
            Post_unzip = dir(fullfile(coregistration_path, sub_raw, 'unzipped', ...
                             [sub_raw, '_', coreg_dummy_list{session_num}, '_acq-petra_run-02_PDw.nii']));

            % Verify all three files were found before proceeding
            if isempty(T1_unzip)
                fprintf('    WARNING: T1 unzipped not found for %s — run unzip step first\n', sub_raw);
                fprintf(fileID, 'WARNING: T1 unzipped not found for %s\n', sub_raw);
                continue
            end
            if isempty(Pre_unzip)
                fprintf('    WARNING: Petra Pre unzipped not found for %s %s\n', sub_raw, ses);
                fprintf(fileID, 'WARNING: Petra Pre unzipped not found for %s %s\n', sub_raw, ses);
                continue
            end
            if isempty(Post_unzip)
                fprintf('    WARNING: Petra Post unzipped not found for %s %s\n', sub_raw, ses);
                fprintf(fileID, 'WARNING: Petra Post unzipped not found for %s %s\n', sub_raw, ses);
                continue
            end

            % [FIX] guard against duplicates here too
            T1_unzip   = T1_unzip(1);
            Pre_unzip  = Pre_unzip(1);
            Post_unzip = Post_unzip(1);

            ref_path        = fullfile(T1_unzip.folder,   T1_unzip.name);
            source_path     = fullfile(Pre_unzip.folder,  Pre_unzip.name);
            additional_path = fullfile(Post_unzip.folder, Post_unzip.name);

            % Sanity check: all three paths must share the same subject prefix
            expected_prefix = sub_raw;
            paths_ok = contains(ref_path,        expected_prefix) && ...
                       contains(source_path,     expected_prefix) && ...
                       contains(additional_path, expected_prefix);

            if ~paths_ok
                fprintf('    ERROR: Subject mismatch in paths for %s %s — skipping\n', sub_raw, ses);
                fprintf(fileID, 'ERROR: Subject mismatch in paths for %s %s\n', sub_raw, ses);
                fprintf(fileID, '  ref        = %s\n', ref_path);
                fprintf(fileID, '  source     = %s\n', source_path);
                fprintf(fileID, '  additional = %s\n', additional_path);
                continue
            end

            try
                fprintf('    Coregistering: %s %s\n', sub_raw, ses);
                fprintf('      ref    : %s\n', ref_path);
                fprintf('      source : %s\n', source_path);
                matlabbatch{1} = create_batch_cor(ref_path, source_path, additional_path);

                % ---- Batch .mat saved to tmp_dir, not the subject folder ----
                coreg_batch = fullfile(tmp_dir, ...
                                       [sub_raw, '_', ses, '_coregistr_batch.mat']);
                save(coreg_batch, 'matlabbatch');

                spm_jobman('run', matlabbatch);

                % [NEW] verify SPM actually wrote both outputs
                if session_coreg_done(coregistration_path, sub_raw, coreg_dummy_list{session_num})
                    fprintf('    Done: %s %s\n', sub_raw, ses);
                    n_coreg_done = n_coreg_done + 1;
                else
                    fprintf('    WARNING: %s %s ran but rsub outputs are incomplete\n', sub_raw, ses);
                    fprintf(fileID, 'WARNING: %s %s incomplete rsub output\n', sub_raw, ses);
                    n_coreg_failed = n_coreg_failed + 1;
                end
            catch ME
                fprintf('    ERROR: %s %s — %s\n', sub_raw, ses, ME.message);
                fprintf(fileID, 'ERROR: %s %s — %s\n', sub_raw, ses, ME.message);
                n_coreg_failed = n_coreg_failed + 1;
            end
        end
    end

else
    fprintf('\n===== COREGISTRATION SKIPPED =====\n');
end

%% ============================================================
%  WRAP UP
%% ============================================================

fprintf('\n============================================================\n');
fprintf('  SUMMARY\n');
fprintf('============================================================\n');
fprintf('  Subjects processed        : %d\n', length(All_Subjects));
fprintf('  Subjects skipped (unzip)  : %d  (already complete)\n', n_sub_unzip_skipped);
fprintf('  Subjects skipped (coreg)  : %d  (already complete)\n', n_sub_coreg_skipped);
fprintf('  Unzip   done / failed     : %d / %d\n', n_unzip_done, n_unzip_failed);
fprintf('  Coreg   done / skipped / failed : %d / %d / %d\n', ...
        n_coreg_done, n_coreg_skipped, n_coreg_failed);
fprintf('============================================================\n');

fprintf(fileID, '\nSUMMARY: subjects=%d unzip_skipped=%d coreg_skipped=%d ', ...
        length(All_Subjects), n_sub_unzip_skipped, n_sub_coreg_skipped);
fprintf(fileID, 'unzip_done=%d unzip_failed=%d coreg_done=%d coreg_skipped=%d coreg_failed=%d\n', ...
        n_unzip_done, n_unzip_failed, n_coreg_done, n_coreg_skipped, n_coreg_failed);

fclose(fileID);
diary off;

fprintf('\nDone.\n');
fprintf('All tmp/log/batch files are in: %s\n', tmp_dir);
fprintf('  Log file : %s\n', log_path);
fprintf('  Diary    : %s\n', diary_file);
fprintf('Delete the _tmp folder to clean up after reviewing.\n');

%% ============================================================
%  LOCAL FUNCTIONS
%% ============================================================

function [SubFolderDir_func, SubFolderNames_func] = get_subdir(act_directory)
    files_sub = dir(act_directory);
    files_sub(~[files_sub.isdir]) = [];
    tf = ismember({files_sub.name}, {'.', '..'});
    files_sub(tf) = [];
    SubFolderNames_func = {files_sub.name};
    SubFolderDir_func   = {files_sub.folder};
end

function d1 = pick_one(d, label, fileID)
% [FIX] Collapse a multi-element dir() result to its first entry.
% Downstream code does `entry.name`, which throws when dir() returned
% more than one match. Returns d unchanged when it has 0 or 1 elements.
    if numel(d) <= 1
        d1 = d;
        return
    end
    d1 = d(1);
    fprintf('  WARNING: %d files match %s — using the first: %s\n', ...
            numel(d), label, d1.name);
    fprintf(fileID, 'WARNING: %d matches for %s — using %s\n', ...
            numel(d), label, d1.name);
    for k = 2:numel(d)
        fprintf(fileID, '    ignored: %s\n', fullfile(d(k).folder, d(k).name));
    end
end

function tf = session_coreg_done(coreg_path, sub_raw, ses_dash)
% [NEW] A session counts as coregistered only when BOTH resliced
% outputs exist. Checking only run-01 (the old behaviour) marked a
% session complete even when the run-02 reslice had failed.
    unz = fullfile(coreg_path, sub_raw, 'unzipped');
    f1  = fullfile(unz, ['r', sub_raw, '_', ses_dash, '_acq-petra_run-01_PDw.nii']);
    f2  = fullfile(unz, ['r', sub_raw, '_', ses_dash, '_acq-petra_run-02_PDw.nii']);
    tf  = (exist(f1, 'file') == 2) && (exist(f2, 'file') == 2);
end

function tf = subject_unzip_done(filelist, subj, coreg_path, all_session)
% [NEW] True when every archive this subject has in XNAT already has a
% matching .nii in derivatives/<sub>/unzipped. Lets the unzip loop skip
% a finished subject without touching the disk per session.
    sub_raw = filelist.(subj).raw_name;
    unz     = fullfile(coreg_path, sub_raw, 'unzipped');
    tf      = false;

    if ~exist(unz, 'dir')
        return
    end

    % T1 from ses-base
    if isfield(filelist.(subj).Session, 'sesbase') && ...
       ~isempty(filelist.(subj).Session.sesbase.T1)
        t1name = filelist.(subj).Session.sesbase.T1.name;   % *.nii.gz
        if exist(fullfile(unz, t1name(1:end-3)), 'file') ~= 2
            return
        end
    end

    % Petra runs from ses-1 .. ses-4  (all_session index 2:end)
    for k = 2:length(all_session)
        ses = all_session{k};
        if ~isfield(filelist.(subj).Session, ses)
            continue
        end
        pre = filelist.(subj).Session.(ses).Petra_Pre;
        if isempty(pre)
            continue
        end
        if exist(fullfile(unz, pre.name(1:end-3)), 'file') ~= 2
            return
        end
        post = filelist.(subj).Session.(ses).Petra_Post;
        if ~isempty(post) && exist(fullfile(unz, post.name(1:end-3)), 'file') ~= 2
            return
        end
    end

    tf = true;
end

function matlabbatch = create_batch_cor(ref_image_path, source_image_path, additional_image_path)
    matlabbatch.spm.spatial.coreg.estwrite.ref    = {[ref_image_path,            ',1']};
    matlabbatch.spm.spatial.coreg.estwrite.source = {[source_image_path,         ',1']};
    matlabbatch.spm.spatial.coreg.estwrite.other  = {[additional_image_path,     ',1']};
    matlabbatch.spm.spatial.coreg.estwrite.eoptions.cost_fun = 'nmi';
    matlabbatch.spm.spatial.coreg.estwrite.eoptions.sep      = [4 2];
    matlabbatch.spm.spatial.coreg.estwrite.eoptions.tol      = [0.02 0.02 0.02 0.001 0.001 0.001 0.01 0.01 0.01 0.001 0.001 0.001];
    matlabbatch.spm.spatial.coreg.estwrite.eoptions.fwhm     = [7 7];
    matlabbatch.spm.spatial.coreg.estwrite.roptions.interp   = 4;
    matlabbatch.spm.spatial.coreg.estwrite.roptions.wrap     = [0 0 0];
    matlabbatch.spm.spatial.coreg.estwrite.roptions.mask     = 0;
    matlabbatch.spm.spatial.coreg.estwrite.roptions.prefix   = 'r';
end
