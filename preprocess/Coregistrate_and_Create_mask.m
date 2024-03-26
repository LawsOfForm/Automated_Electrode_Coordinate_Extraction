%% create struct for files 

% spm('defaults','fmri');
% spm_jobman('initcfg');

% get als subjects in list
%path = "/media/Data03/Studies/MeMoSLAP/SUBJECTS";
path = "/media/MeMoSLAP_Subjects/SUBJECTS_XNAT";
[SubFolderDir, SubFolderNames] = get_subdir(path);

%segment all subjects from all locations
for subfolder_subject=1:length(SubFolderNames)
      %if (length(SubFolderNames{subfolder_subject}) == 7 || length(SubFolderNames{subfolder_subject}) == 8) % subfolder must have this length otherwise wrong folder
      if (length(SubFolderNames{subfolder_subject}) == 7) % subfolder must have this length otherwise wrong folder
  
          SubFolderNames{subfolder_subject}=SubFolderNames{subfolder_subject};
          SubFolderDir{subfolder_subject}=SubFolderDir{subfolder_subject};
      else
          SubFolderNames{subfolder_subject}={};
          SubFolderDir{subfolder_subject}={};
      end
end

SubFolderNames=SubFolderNames(~cellfun('isempty',SubFolderNames));
SubFolderDir=SubFolderDir(~cellfun('isempty',SubFolderDir));

%filelist.SubFolderNames = SubFolderNames
% get all sessions in list
for subfolder_subject=1:length(SubFolderNames)
    %coregistration_path="/media/Data03/Studies/MeMoSLAP/derivatives/coregistration_coordinates";
    coregistration_path="/media/MeMoSLAP_Subjects/derivatives/automated_electrode_extraction";
        fprintf('Sub folder #%d = %s\n', subfolder_subject, SubFolderNames{subfolder_subject});
        path = fullfile(SubFolderDir{subfolder_subject}, SubFolderNames{subfolder_subject});
        [SesFolderDir, SesFolderNames] = get_subdir(path);
        sub_name = erase(SubFolderNames{subfolder_subject},"-");
        filelist.(sub_name).Session_list = SesFolderNames;
    
        %%% loop throught Session for each Subject
        % get all files in every session
        for subfolder_session = 1:length(SesFolderNames)
    
            ses_name = erase(SesFolderNames{subfolder_session},"-");
            path=fullfile(SesFolderDir{subfolder_session}, SesFolderNames{subfolder_session});
            
            filelist.(sub_name).Session.(ses_name).T1 = dir(fullfile(path, '**/*mprage_T1w.nii.gz'));
            filelist.(sub_name).Session.(ses_name).Petra_Pre = dir(fullfile(path, '**/*petra_run-01_PDw.nii.gz'));
            filelist.(sub_name).Session.(ses_name).Petra_Post = dir(fullfile(path, '**/*petra_run-02_PDw.nii.gz'));
        end       
end

log_file='Logfile_Coregistration.txt';
% Open a file for writing (or appending) using fopen
%fileID = fopen('logfile.txt', 'a'); % 'a' for append mode
fileID = fopen(fullfile(coregistration_path, log_file),'w'); % 'a' for append mode

if fileID == -1
    error('Could not open file for writing');
end

dummy_folder_list={'sesbase','ses-1','ses-2','ses-3','ses-4'};
all_session={'sesbase','ses1','ses2','ses3','ses4'};
All_Subjects=fieldnames(filelist);
%% loop through struct and create unzipped files
%uncomment this part if you only want to coregister
for sub =1:length(All_Subjects)
        %all_session = fieldnames(filelist.(All_Subjects{sub}).Session);
        for session_num = 1:length(all_session)
            if ((isfield(filelist.(All_Subjects{sub}).Session, "sesbase")) && all_session{session_num}=="sesbase")
                %% sesbase T1 unzip will only be in sesbase
                    %try
                    if exist("T1_path","var")
                       clear T1_path
                    end
                    if ~isempty(filelist.(All_Subjects{sub}).Session.(all_session{session_num}).T1)
                        T1_folder=fullfile(filelist.(All_Subjects{sub}).Session.(all_session{session_num}).T1.folder);
                        T1_name = fullfile(filelist.(All_Subjects{sub}).Session.(all_session{session_num}).T1.name);
                        T1_path = fullfile(T1_folder, T1_name);
                        T1_path_nii=fullfile(coregistration_path, SubFolderNames{sub}, 'unzipped', T1_name(1:end-3));
                    


                        if exist(T1_path,'file') && ...
                        ~exist(T1_path_nii,'file') 
                        fprintf('T1 unzipping running subject %s at Session %s \n', All_Subjects{sub}, all_session{session_num})
                        gunzip(T1_path,fullfile(coregistration_path, SubFolderNames{sub}, 'unzipped'))
                        filelist.(All_Subjects{sub}).Session.(all_session{session_num}).T1_unzip = dir(fullfile(coregistration_path, SubFolderNames{sub}, 'unzipped', [SubFolderNames{sub},'_ses-base_acq-mprage_T1w.nii']));
                        end
                    else
                        fprintf('T1 not found for subject %s at Session %s \n', All_Subjects{sub}, all_session{session_num})
                        fprintf(fileID, 'T1 not found for subject %s at Session %s \n', All_Subjects{sub}, all_session{session_num});
                    end
                    %catch
                    %    fprintf('something went wrong unzipping T1 subject %s at Session %s \n', All_Subjects{sub}, all_session{session_num})
                    %end
                    
            end
                    %% all other sessions contain Petra_Post and Petra_Pre unziped files
            if  isfield(filelist.(All_Subjects{sub}).Session, "sesbase") && all_session{session_num} ~= "sesbase" && ...
                ((isfield(filelist.(All_Subjects{sub}).Session,'ses1') && all_session{session_num}=="ses1" )|| ...  
                (isfield(filelist.(All_Subjects{sub}).Session,'ses2') && all_session{session_num}=="ses2" ) || ...  
                (isfield(filelist.(All_Subjects{sub}).Session,'ses3') && all_session{session_num}=="ses3" ) || ...  
                (isfield(filelist.(All_Subjects{sub}).Session,'ses4') && all_session{session_num}=="ses4" ))  

                %try  % there might be corrupted folder
                if exist("Petra_Pre_path","var")
                   clear Petra_Pre_path
                end

                if ~isempty(filelist.(All_Subjects{sub}).Session.(all_session{session_num}).Petra_Pre) 
                Petra_Pre_folder=fullfile(filelist.(All_Subjects{sub}).Session.(all_session{session_num}).Petra_Pre.folder);
                Petra_Pre_name=fullfile(filelist.(All_Subjects{sub}).Session.(all_session{session_num}).Petra_Pre.name);
                Petra_Pre_path=fullfile(Petra_Pre_folder, Petra_Pre_name);
                Petra_Pre_path_nii=fullfile(coregistration_path, SubFolderNames{sub}, 'unzipped', Petra_Pre_name(1:end-3));


                    if exist(Petra_Pre_path,'file') && ...
                      ~exist(Petra_Pre_path_nii,'file') %delete .gz from nii.gz file appendix
                    fprintf('gunzip running for Petra subjectct %s at Session %s \n', All_Subjects{sub}, all_session{session_num})
                    gunzip(fullfile(filelist.(All_Subjects{sub}).Session.(all_session{session_num}).Petra_Pre.folder,filelist.(All_Subjects{sub}).Session.(all_session{session_num}).Petra_Pre.name),fullfile(coregistration_path, SubFolderNames{sub}, 'unzipped'))
                    gunzip(fullfile(filelist.(All_Subjects{sub}).Session.(all_session{session_num}).Petra_Post.folder,filelist.(All_Subjects{sub}).Session.(all_session{session_num}).Petra_Post.name),fullfile(coregistration_path, SubFolderNames{sub}, 'unzipped'))
                    
                    filelist.(All_Subjects{sub}).Session.(all_session{session_num}).Petra_Pre_unzip = dir(fullfile(coregistration_path, SubFolderNames{sub}, 'unzipped', [SubFolderNames{sub},'_',dummy_folder_list{session_num},'_','acq-petra_run-01_PDw.nii']));
                    filelist.(All_Subjects{sub}).Session.(all_session{session_num}).Petra_Post_unzip = dir(fullfile(coregistration_path, SubFolderNames{sub}, 'unzipped', [SubFolderNames{sub},'_',dummy_folder_list{session_num},'_','acq-petra_run-02_PDw.nii']));
                    end
                else 
                    fprintf('something wrong with Petra subject %s at Session %s maybe just one petra in folder\n', All_Subjects{sub}, all_session{session_num})

                    fprintf(fileID,'something wrong with Petra subject %s at Session %s maybe just one petra in folder\n', All_Subjects{sub}, all_session{session_num});
                end
                    %catch
                %fprintf('something wrong for Petra conversion subject %s at Session %s \n', All_Subjects{sub}, all_session{session_num})
                %end
             end
        end
end

%% comment above part if you just want to segment and coregister, unzipping will take a while

%% loop through all subjects and create matlabbatch
for sub =1:length(All_Subjects)
    
  
        %% loop through all sessions
  
            if isfield(filelist.(All_Subjects{sub}).Session, "sesbase")
                all_session = {'ses1','ses2','ses3','ses4'};
                dummy_folder_list={'ses-1','ses-2','ses-3','ses-4'};
                for session_num = 1:length(all_session)
                    if isfield(filelist.(All_Subjects{sub}).Session, all_session(session_num)) && ...
                       ~isempty(filelist.(All_Subjects{sub}).Session.(all_session{session_num}).Petra_Pre) && ...
                       ~isempty(filelist.(All_Subjects{sub}).Session.sesbase.T1)    
                            
                        try
                        filelist.(All_Subjects{sub}).Session.sesbase.T1_unzip = dir(fullfile(coregistration_path, SubFolderNames{sub}, 'unzipped', [SubFolderNames{sub},'_ses-base_acq-mprage_T1w.nii']));
                        filelist.(All_Subjects{sub}).Session.(all_session{session_num}).Petra_Pre_unzip = dir(fullfile(coregistration_path, SubFolderNames{sub}, 'unzipped', [SubFolderNames{sub},'_',dummy_folder_list{session_num},'_','acq-petra_run-01_PDw.nii']));
                        filelist.(All_Subjects{sub}).Session.(all_session{session_num}).Petra_Post_unzip = dir(fullfile(coregistration_path, SubFolderNames{sub}, 'unzipped', [SubFolderNames{sub},'_',dummy_folder_list{session_num},'_','acq-petra_run-02_PDw.nii']));
 

                        ref_image_folder = filelist.(All_Subjects{sub}).Session.sesbase.T1_unzip.folder;
                        ref_image_name = filelist.(All_Subjects{sub}).Session.sesbase.T1_unzip.name;
                        ref_image_path = fullfile(ref_image_folder, ref_image_name);
                        source_image_folder = filelist.(All_Subjects{sub}).Session.(all_session{session_num}).Petra_Pre_unzip.folder;
                        source_image_name = filelist.(All_Subjects{sub}).Session.(all_session{session_num}).Petra_Pre_unzip.name;
                        source_image_path = fullfile(source_image_folder, source_image_name);
                        %source_image_path_cor=fullfile(source_image_folder, ['r',source_image_name,'.gz']);
                        source_image_path_cor=fullfile(source_image_folder, ['r',source_image_name]);
                        
                        additional_image_folder = filelist.(All_Subjects{sub}).Session.(all_session{session_num}).Petra_Post_unzip.folder;
                        additional_image_name = filelist.(All_Subjects{sub}).Session.(all_session{session_num}).Petra_Post_unzip.name;
                        additional_image_path = fullfile(additional_image_folder, additional_image_name);
                        %%create batch file in loop is sesbase exists
                        catch
                            fprintf('Coregistration Path problem with session %s for subject %s\n', all_session{session_num},All_Subjects{sub})
                       
                            fprintf(fileID,'Coregistration Path problem with session %s for subject %s\n', all_session{session_num},All_Subjects{sub});
                        end

                        if exist("source_image_path_cor","var") && ~exist(source_image_path_cor,'file')  

                            try
                            fprintf('Coregistration for session %s for subject %s\n', all_session{session_num},All_Subjects{sub})
                            coregistr_batch = fullfile(coregistration_path, SubFolderNames{sub},[all_session{session_num},'coregistr_batch.mat']);
                            %matlabbatch{session_num} = create_batch_cor(ref_image_path, source_image_path,additional_image_path);
                            matlabbatch{1} = create_batch_cor(ref_image_path, source_image_path,additional_image_path);
                            save(coregistr_batch,'matlabbatch');
                            spm_jobman('run',matlabbatch);
                            catch
                            fprintf('Coregistration error in session %s for subject %s\n', all_session{session_num},All_Subjects{sub})
                            fprintf(fileID,'Coregistration error in session %s for subject %s\n', all_session{session_num},All_Subjects{sub});
                            end
                        else
                        fprintf('Corregistration already done for session %s for subject %s\n', all_session{session_num},All_Subjects{sub})    
                        end
                    else 
                        fprintf('No session %s for subject %s\n', all_session{session_num},All_Subjects{sub})
                        fprintf(fileID,'No session %s for subject %s\n', all_session{session_num},All_Subjects{sub});
                    
                    end

                end
                try
                coregistr_batch = fullfile(coregistration_path, SubFolderNames{sub},'coregistr_batch.mat');
                    if ~exist(coregistr_batch,'file')
                    %if ~exist(coregistr_batch,'file')
                    save(coregistr_batch,'matlabbatch');
                    spm_jobman('run',matlabbatch);
                    end
                catch
                fprintf('Corregistration already done for all known sessions subject %s\n',All_Subjects{sub})
                end
            end
end


%% create segmentation for all T1 files
for sub =1:length(All_Subjects)
    
    if (exist(fullfile(coregistration_path, SubFolderNames{sub},'unzipped'),'dir') && ...
        isfield(filelist.(All_Subjects{sub}).Session, "sesbase") && ...
        ~exist(fullfile(coregistration_path, SubFolderNames{sub},'unzipped',['c1',SubFolderNames{sub},'_ses-base_acq-mprage_T1w.nii']),'file')) && ...
        ~isempty(filelist.(All_Subjects{sub}).Session.sesbase.T1)
        filelist.(All_Subjects{sub}).Session.sesbase.T1_unzip = dir(fullfile(coregistration_path, SubFolderNames{sub}, 'unzipped', [SubFolderNames{sub},'*mprage_T1w.nii']));

        
        ref_image_folder = filelist.(All_Subjects{sub}).Session.sesbase.T1_unzip.folder;
        ref_image_name = filelist.(All_Subjects{sub}).Session.sesbase.T1_unzip.name;
        ref_image_path = fullfile(ref_image_folder, ref_image_name);

        matlabbatch{1} = create_batch_segm(ref_image_path);
        matlabbatch{2} = create_batch_image_calculator(ref_image_folder,SubFolderNames{sub});

        coregistr_batch = fullfile(coregistration_path, SubFolderNames{sub},'coregistr_batch_seg_calc.mat');
        save(coregistr_batch,'matlabbatch');
        spm_jobman('run',matlabbatch);




        
    else       
        fprintf('segmentation alread done for subject %s\n',All_Subjects{sub})
    end
end
  

   


function [SubFolderDir_func, SubFolderNames_func] = get_subdir(act_directory)
files_sub = dir(act_directory);
files_sub(~[files_sub.isdir])=[];
tf = ismember({files_sub.name}, {'.', '..'});
files_sub(tf) = [];  %remove current and parent directory.

SubFolderNames_func = {files_sub.name};
SubFolderDir_func = {files_sub.folder};

end


function [matlabbatch] = create_batch_cor(ref_image_path, source_image_path, additional_image_path)


matlabbatch.spm.spatial.coreg.estwrite.ref = {[ref_image_path,',1']};
matlabbatch.spm.spatial.coreg.estwrite.source = {[source_image_path,',1']};
matlabbatch.spm.spatial.coreg.estwrite.other = {[additional_image_path,',1']};
matlabbatch.spm.spatial.coreg.estwrite.eoptions.cost_fun = 'nmi';
matlabbatch.spm.spatial.coreg.estwrite.eoptions.sep = [4 2];
matlabbatch.spm.spatial.coreg.estwrite.eoptions.tol = [0.02 0.02 0.02 0.001 0.001 0.001 0.01 0.01 0.01 0.001 0.001 0.001];
matlabbatch.spm.spatial.coreg.estwrite.eoptions.fwhm = [7 7];
matlabbatch.spm.spatial.coreg.estwrite.roptions.interp = 4;
matlabbatch.spm.spatial.coreg.estwrite.roptions.wrap = [0 0 0];
matlabbatch.spm.spatial.coreg.estwrite.roptions.mask = 0;
matlabbatch.spm.spatial.coreg.estwrite.roptions.prefix = 'r';

end

function [matlabbatch] = create_batch_segm(ref_image_path)



matlabbatch.spm.spatial.preproc.channel.vols = {[ref_image_path,',1']};
matlabbatch.spm.spatial.preproc.channel.biasreg = 0.001;
matlabbatch.spm.spatial.preproc.channel.biasfwhm = 60;
matlabbatch.spm.spatial.preproc.channel.write = [0 0];
matlabbatch.spm.spatial.preproc.tissue(1).tpm = {'/home/niemannf/Documents/MATLAB/spm12/tpm/TPM.nii,1'};
matlabbatch.spm.spatial.preproc.tissue(1).ngaus = 1;
matlabbatch.spm.spatial.preproc.tissue(1).native = [1 0];
matlabbatch.spm.spatial.preproc.tissue(1).warped = [0 0];
matlabbatch.spm.spatial.preproc.tissue(2).tpm = {'/home/niemannf/Documents/MATLAB/spm12/tpm/TPM.nii,2'};
matlabbatch.spm.spatial.preproc.tissue(2).ngaus = 1;
matlabbatch.spm.spatial.preproc.tissue(2).native = [1 0];
matlabbatch.spm.spatial.preproc.tissue(2).warped = [0 0];
matlabbatch.spm.spatial.preproc.tissue(3).tpm = {'/home/niemannf/Documents/MATLAB/spm12/tpm/TPM.nii,3'};
matlabbatch.spm.spatial.preproc.tissue(3).ngaus = 2;
matlabbatch.spm.spatial.preproc.tissue(3).native = [1 0];
matlabbatch.spm.spatial.preproc.tissue(3).warped = [0 0];
matlabbatch.spm.spatial.preproc.tissue(4).tpm = {'/home/niemannf/Documents/MATLAB/spm12/tpm/TPM.nii,4'};
matlabbatch.spm.spatial.preproc.tissue(4).ngaus = 3;
matlabbatch.spm.spatial.preproc.tissue(4).native = [1 0];
matlabbatch.spm.spatial.preproc.tissue(4).warped = [0 0];
matlabbatch.spm.spatial.preproc.tissue(5).tpm = {'/home/niemannf/Documents/MATLAB/spm12/tpm/TPM.nii,5'};
matlabbatch.spm.spatial.preproc.tissue(5).ngaus = 4;
matlabbatch.spm.spatial.preproc.tissue(5).native = [1 0];
matlabbatch.spm.spatial.preproc.tissue(5).warped = [0 0];
matlabbatch.spm.spatial.preproc.tissue(6).tpm = {'/home/niemannf/Documents/MATLAB/spm12/tpm/TPM.nii,6'};
matlabbatch.spm.spatial.preproc.tissue(6).ngaus = 2;
matlabbatch.spm.spatial.preproc.tissue(6).native = [0 0];
matlabbatch.spm.spatial.preproc.tissue(6).warped = [0 0];
matlabbatch.spm.spatial.preproc.warp.mrf = 1;
matlabbatch.spm.spatial.preproc.warp.cleanup = 1;
matlabbatch.spm.spatial.preproc.warp.reg = [0 0.001 0.5 0.05 0.2];
matlabbatch.spm.spatial.preproc.warp.affreg = 'mni';
matlabbatch.spm.spatial.preproc.warp.fwhm = 0;
matlabbatch.spm.spatial.preproc.warp.samp = 3;
matlabbatch.spm.spatial.preproc.warp.write = [0 0];
matlabbatch.spm.spatial.preproc.warp.vox = NaN;
matlabbatch.spm.spatial.preproc.warp.bb = [NaN NaN NaN
                                              NaN NaN NaN];

end

function [matlabbatch] = create_batch_image_calculator(path, sub)


matlabbatch.spm.util.imcalc.input = {
                                       fullfile(path,['/c1',sub,'_ses-base_acq-mprage_T1w.nii,1'])
                                        fullfile(path,['/c2',sub,'_ses-base_acq-mprage_T1w.nii,1'])
                                        fullfile(path,['/c3',sub,'_ses-base_acq-mprage_T1w.nii,1'])
                                        fullfile(path,['/c4',sub,'_ses-base_acq-mprage_T1w.nii,1'])
                                        fullfile(path,['/c5',sub,'_ses-base_acq-mprage_T1w.nii,1'])
                                        };
matlabbatch.spm.util.imcalc.output = 'finalmask_SPM';
matlabbatch.spm.util.imcalc.outdir = {fullfile(path)};
matlabbatch.spm.util.imcalc.expression = 'i1+i2+i3+i4+i5';
matlabbatch.spm.util.imcalc.var = struct('name', {}, 'value', {});
matlabbatch.spm.util.imcalc.options.dmtx = 0;
matlabbatch.spm.util.imcalc.options.mask = 0;
matlabbatch.spm.util.imcalc.options.interp = 1;
matlabbatch.spm.util.imcalc.options.dtype = 4;

end
%clear
%clc
